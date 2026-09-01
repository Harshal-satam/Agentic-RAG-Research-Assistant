
import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi


PDF_PATH = r"C:\Users\lenovo\Desktop\Agentic-RAG-Research-Assistant\data\research_papers\r1.pdf"


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

def load_documents():

    pdf = fitz.open(PDF_PATH)

    documents = []

    for page_number, page in enumerate(pdf, 1):

        text = page.get_text()

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "r1.pdf",
                        "page": page_number
                    }
                )
            )

    pdf.close()

    return documents


# --------------------------------------------------
# CHUNK DOCUMENTS
# --------------------------------------------------

def create_chunks():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks, 1):

        chunk.metadata["chunk_id"] = i

    return chunks


# --------------------------------------------------
# BUILD RETRIEVERS
# --------------------------------------------------

def build_retrievers():

    chunks = create_chunks()

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    tokenized_corpus = [
        doc.page_content.lower().split()
        for doc in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return chunks, vector_store, bm25


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def hybrid_search(
    query,
    vector_store,
    bm25,
    chunks,
    k=3,
    alpha=0.5
):

    semantic_docs = vector_store.similarity_search(
        query,
        k=k * 2
    )

    scores = bm25.get_scores(
        query.lower().split()
    )

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k * 2]

    keyword_docs = [
        chunks[i]
        for i in top_indices
    ]

    combined_scores = {}
    doc_lookup = {}

    # FAISS results
    for rank, doc in enumerate(semantic_docs):

        doc_id = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id")
        )

        combined_scores[doc_id] = (
            combined_scores.get(doc_id, 0)
            + alpha * (1 / (rank + 1))
        )

        doc_lookup[doc_id] = doc

    # BM25 results
    for rank, doc in enumerate(keyword_docs):

        doc_id = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id")
        )

        combined_scores[doc_id] = (
            combined_scores.get(doc_id, 0)
            + (1 - alpha) * (1 / (rank + 1))
        )

        doc_lookup[doc_id] = doc

    ranked_ids = sorted(
        combined_scores,
        key=combined_scores.get,
        reverse=True
    )[:k]

    return [
        doc_lookup[doc_id]
        for doc_id in ranked_ids
    ]
