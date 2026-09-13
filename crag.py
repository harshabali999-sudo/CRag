import os
import sys
import hashlib
from typing import TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END

# Ensure standard output can print Unicode characters (e.g. math symbols, Greek letters)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables (.env)
load_dotenv()

# -------------------------------------------------------------------------
# 1. LLM Setup (Groq)
# -------------------------------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------------------------------------
# 2. Embedding Model & Vector Store (Chroma)
# -------------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

def generate_id(text: str) -> str:
    """Generate MD5 hash ID for document content to avoid duplicate ingestion."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def add_new_documents(vectorstore: Chroma, documents: List[Document]):
    """Ingest new documents avoiding duplicates by ID."""
    new_documents = []
    new_ids = []

    for doc in documents:
        doc_id = generate_id(doc.page_content)
        existing = vectorstore.get(ids=[doc_id])

        if not existing["ids"]:
            new_documents.append(doc)
            new_ids.append(doc_id)

    if new_documents:
        vectorstore.add_documents(
            documents=new_documents,
            ids=new_ids
        )
        print(f"Added {len(new_documents)} new chunks to vector store.")
    else:
        print("Vector store up to date. No new documents to add.")

# Index PDF if available
pdf_path = os.path.join("Data", "RAG.pdf")
if os.path.exists(pdf_path):
    document_loader = PyPDFLoader(pdf_path)
    raw_docs = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=150,
        chunk_size=400
    )
    doc_chunks = text_splitter.split_documents(raw_docs)
    add_new_documents(vectorstore, doc_chunks)

# Retriever setup
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -------------------------------------------------------------------------
# 3. External Search Client (Tavily)
# -------------------------------------------------------------------------
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# -------------------------------------------------------------------------
# 4. State Definition for CRAG (LangGraph)
# -------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_documents: List[Document]
    scores: List[float]
    action: str
    evaluation: str
    refined_documents: List[Document]
    answer: str

# -------------------------------------------------------------------------
# 5. Document Relevance Evaluator
# -------------------------------------------------------------------------
class RetrievalScore(BaseModel):
    score: float = Field(
        description="Relevance score between -1.0 (completely irrelevant) and 1.0 (highly relevant)"
    )

evaluator = llm.with_structured_output(RetrievalScore)

evaluation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a retrieval relevance evaluator for a Corrective RAG system.

Your task is to evaluate how relevant the retrieved document is
for answering the user's question.

Return a score between -1 and 1.

Score meaning:
 1.0  = highly relevant and directly useful
 0.5  = moderately relevant
 0.0  = uncertain or partially relevant
-0.5  = mostly irrelevant
-1.0  = completely irrelevant

Do not evaluate whether the answer itself is correct.
Only evaluate whether the document contains useful information
for answering the question.

Return ONLY the structured score."""
    ),
    (
        "human",
        """Question:
{question}

Retrieved Document:
{document}"""
    )
])

evaluator_chain = evaluation_prompt | evaluator

# CRAG Confidence Thresholds
UPPER_THRESHOLD = 0.70
LOWER_THRESHOLD = -0.70

# -------------------------------------------------------------------------
# 6. Graph Nodes
# -------------------------------------------------------------------------
def retrieve(state: GraphState):
    """Retrieve relevant documents from internal vector database."""
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def evaluate_documents(state: GraphState):
    """Evaluate relevance score of retrieved documents and determine CRAG action."""
    question = state["question"]
    documents = state.get("documents", [])

    scores = []
    for document in documents:
        result = evaluator_chain.invoke({
            "question": question,
            "document": document.page_content
        })
        scores.append(result.score)

    if not scores:
        action = "incorrect"
    else:
        max_score = max(scores)
        if max_score >= UPPER_THRESHOLD:
            action = "correct"
        elif max_score <= LOWER_THRESHOLD:
            action = "incorrect"
        else:
            action = "ambiguous"

    print(f"[evaluate_documents] Scores: {scores} -> Action: '{action}'")
    return {
        "scores": scores,
        "action": action,
        "evaluation": action
    }

def decide_action(state: GraphState) -> str:
    """Conditional routing function based on evaluated action."""
    action = state.get("action")
    if action:
        return action

    # Fallback if action wasn't set
    scores = state.get("scores", [])
    if not scores:
        return "incorrect"
    max_score = max(scores)
    if max_score >= UPPER_THRESHOLD:
        return "correct"
    elif max_score <= LOWER_THRESHOLD:
        return "incorrect"
    else:
        return "ambiguous"

def web_search(state: GraphState):
    """Perform web search using Tavily client if retrieval was incorrect or ambiguous."""
    question = state["question"]
    print(f"[web_search] Searching web for: '{question}'...")

    response = tavily_client.search(
        query=question,
        search_depth="advanced",
        max_results=4
    )

    web_documents = []
    for result in response.get("results", []):
        doc = Document(
            page_content=result.get("content", ""),
            metadata={
                "source": result.get("url", ""),
                "title": result.get("title", "")
            }
        )
        web_documents.append(doc)

    return {"web_documents": web_documents}

def knowledge_refinement(state: GraphState):
    """Refine and filter knowledge before generation based on the CRAG action."""
    action = state.get("action", "ambiguous")
    question = state["question"]

    if action == "correct":
        documents = state.get("documents", [])
    elif action == "ambiguous":
        # In CRAG, ambiguous combines both internal documents and external web knowledge
        documents = (state.get("documents") or []) + (state.get("web_documents") or [])
    else:  # "incorrect"
        documents = state.get("web_documents") or []

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = f"""You are the knowledge refinement component of a Corrective RAG system.

Question:
{question}

Retrieved knowledge:
{context}

Your task is to refine the retrieved knowledge:
1. Keep information relevant to the question.
2. Remove irrelevant information.
3. Remove duplicated information.
4. Preserve important factual details.
5. Do not add information that is not present in the retrieved knowledge.

Return only the refined knowledge."""

    response = llm.invoke(prompt)
    refined_document = Document(page_content=response.content)

    return {"refined_documents": [refined_document]}

def generate_answer(state: GraphState):
    """Generate the final grounded answer from refined knowledge."""
    question = state["question"]
    documents = state.get("refined_documents", [])

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = f"""Answer the user's question using the provided context.

Question:
{question}

Context:
{context}

Use only the information available in the context.
If the context does not contain enough information, say that you do not have enough information.
Give a clear and accurate answer."""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# -------------------------------------------------------------------------
# 7. Construct & Compile the State Graph
# -------------------------------------------------------------------------
graph_builder = StateGraph(GraphState)

# Nodes
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("evaluate_documents", evaluate_documents)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("knowledge_refinement", knowledge_refinement)
graph_builder.add_node("generate_answer", generate_answer)

# START -> retrieve -> evaluate_documents
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "evaluate_documents")

# Evaluate -> Conditional Decision
graph_builder.add_conditional_edges(
    "evaluate_documents",
    decide_action,
    {
        "correct": "knowledge_refinement",
        "incorrect": "web_search",
        "ambiguous": "web_search"
    }
)

# Web Search -> Knowledge Refinement
graph_builder.add_edge("web_search", "knowledge_refinement")

# Knowledge Refinement -> Generate Answer
graph_builder.add_edge("knowledge_refinement", "generate_answer")

# Generate Answer -> END
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()

# -------------------------------------------------------------------------
# 8. Execution Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    test_question = "What is gradient descent?"
    initial_state = {
        "question": test_question,
        "documents": [],
        "web_documents": [],
        "scores": [],
        "action": "",
        "evaluation": "",
        "refined_documents": [],
        "answer": ""
    }

    print(f"\n--- Running CRAG for: '{test_question}' ---")
    result = graph.invoke(initial_state)

    print("\n================ FINAL RESULT ================")
    print(f"Action Taken: {result.get('action')}")
    print(f"Evaluated Scores: {result.get('scores')}")
    print(f"\nAnswer:\n{result.get('answer')}")
