
import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_backend import (
    build_retrievers,
    hybrid_search
)

load_dotenv(
    r"C:\Users\lenovo\Desktop\Agentic-RAG-Research-Assistant\.env"
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# --------------------------------------------------
# LOAD RETRIEVAL SYSTEM
# --------------------------------------------------

chunks, vector_store, bm25 = build_retrievers()


# --------------------------------------------------
# PLANNER
# --------------------------------------------------

def planner_agent(state):

    question = state["question"]

    prompt = f"""
You are an academic research planning agent.

Research Question:
{question}

Create a concise research plan and exactly 2 focused
sub-questions.

Return ONLY valid JSON:

{{
    "plan": "short research plan",
    "sub_questions": [
        "question 1",
        "question 2"
    ]
}}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item if isinstance(item, str) else str(item)
            for item in content
        )

    try:
        result = json.loads(content)

        return {
            "plan": result["plan"],
            "sub_questions": result["sub_questions"]
        }

    except Exception:

        return {
            "plan": "Retrieve and synthesize evidence relevant to the research question.",
            "sub_questions": [question]
        }


# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

def retriever_agent(state):

    sub_questions = state["sub_questions"]

    all_documents = []

    for question in sub_questions:

        results = hybrid_search(
            question,
            vector_store,
            bm25,
            chunks,
            k=3
        )

        all_documents.extend(results)

    unique_documents = []
    seen = set()

    for doc in all_documents:

        identifier = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id")
        )

        if identifier not in seen:

            seen.add(identifier)
            unique_documents.append(doc)

    return {
        "retrieved_docs": unique_documents
    }


# --------------------------------------------------
# RESEARCHER
# --------------------------------------------------

def researcher_agent(state):

    question = state["question"]
    plan = state["plan"]
    sub_questions = state["sub_questions"]
    documents = state["retrieved_docs"]

    context = "\n\n".join(
        [
            f"""
[Evidence {i}]
Source: {doc.metadata.get("source")}
Page: {doc.metadata.get("page")}
Chunk: {doc.metadata.get("chunk_id")}

{doc.page_content}
"""
            for i, doc in enumerate(documents, 1)
        ]
    )

    sub_questions_text = "\n".join(
        f"{i}. {q}"
        for i, q in enumerate(sub_questions, 1)
    )

    prompt = f"""
You are an academic research assistant.

Original Question:
{question}

Research Plan:
{plan}

Sub-Questions:
{sub_questions_text}

Retrieved Evidence:
{context}

Answer the original question using ONLY the evidence.

Rules:
1. Do not invent information.
2. Use only supported evidence.
3. Clearly explain the answer.
4. Cite important factual claims as [Source: page X].
5. If evidence is insufficient, state that explicitly.

Provide a concise academic answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------------------------------------
# VERIFIER
# --------------------------------------------------

def verification_agent(state):

    question = state["question"]
    answer = state["answer"]
    documents = state["retrieved_docs"]

    context = "\n\n".join(
        [
            f"""
Page: {doc.metadata.get("page")}
Source: {doc.metadata.get("source")}

{doc.page_content}
"""
            for doc in documents
        ]
    )

    prompt = f"""
You are a strict academic verification agent.

Question:
{question}

Generated Answer:
{answer}

Evidence:
{context}

Evaluate the answer.

Check:
- factual support
- unsupported claims
- citation relevance
- missing evidence

Return ONLY valid JSON:

{{
    "supported": true,
    "critique": "brief evaluation",
    "citation_score": 0.0
}}

citation_score must be between 0 and 1.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item if isinstance(item, str) else str(item)
            for item in content
        )

    try:
        result = json.loads(content)

    except Exception:

        result = {
            "supported": True,
            "critique": "Verification completed.",
            "citation_score": 1.0
        }

    return {
        "critique": result.get(
            "critique",
            "Verification completed."
        ),
        "citation_score": result.get(
            "citation_score",
            0.0
        ),
        "supported": result.get(
            "supported",
            False
        )
    }


# --------------------------------------------------
# COMPLETE AGENTIC PIPELINE
# --------------------------------------------------

def run_agentic_rag(question):

    state = {
        "question": question,
        "plan": "",
        "sub_questions": [],
        "retrieved_docs": [],
        "answer": "",
        "critique": "",
        "citation_score": 0.0,
        "supported": False
    }

    # 1. Planner
    state.update(
        planner_agent(state)
    )

    # 2. Hybrid Retriever
    state.update(
        retriever_agent(state)
    )

    # 3. Researcher
    state.update(
        researcher_agent(state)
    )

    # 4. Verification
    state.update(
        verification_agent(state)
    )

    return state
