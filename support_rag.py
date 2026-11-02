import os
import logging
import requests

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:0.5b"
)

# LLM backend: "ollama" (local) or "groq" (hosted, for cloud deploys).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a customer support assistant. "
    "Answer questions using ONLY the supplied "
    "knowledge-base context. "
    "If the context does not contain the answer, "
    "say: Insufficient information in the "
    "knowledge base to answer this question. "
    "Do not invent policies, prices, or facts."
)

# Sample customer support knowledge base.
# Replace these records with your Experiment 5/6 documents
# when integrating your existing RAG pipeline.

KNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "Account Registration",
        "content": (
            "Customers can create an account using their email address. "
            "They must provide the required registration information "
            "and verify their email address."
        )
    },
    {
        "id": 2,
        "title": "Password Reset",
        "content": (
            "Customers who forget their password can use the "
            "Forgot Password option on the login page. "
            "A password reset link is sent to their registered email."
        )
    },
    {
        "id": 3,
        "title": "Order Tracking",
        "content": (
            "Customers can track an order using the tracking link "
            "provided in their order confirmation email. "
            "Tracking information depends on the latest shipping update."
        )
    },
    {
        "id": 4,
        "title": "Refund Policy",
        "content": (
            "Customers should contact customer support with their "
            "order ID to enquire about refund eligibility. "
            "The applicable refund conditions must be checked "
            "before a refund can be approved."
        )
    },
    {
        "id": 5,
        "title": "Customer Support",
        "content": (
            "Customers can contact the customer support team "
            "through the support section of the application. "
            "They should provide their order ID when asking "
            "for help with an order."
        )
    }
]


def retrieve_documents(question: str, top_k: int = 3):
    """
    Retrieve the most relevant knowledge-base documents
    using a simple keyword-overlap scoring method.
    """

    stop_words = {
        "the", "is", "a", "an", "to", "for", "of",
        "in", "on", "and", "or", "i", "my", "me",
        "can", "how", "what", "do", "does", "please",
        "tell", "about", "with", "your", "you"
    }

    question_words = {
        word.lower().strip(".,?!:")
        for word in question.split()
        if word.lower().strip(".,?!:") not in stop_words
    }

    scored_documents = []

    for document in KNOWLEDGE_BASE:
        document_text = (
            document["title"] + " " + document["content"]
        ).lower()

        document_words = set(
            word.strip(".,?!:")
            for word in document_text.split()
        )

        score = len(question_words & document_words)

        if score > 0:
            scored_documents.append((score, document))

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        document
        for score, document in scored_documents[:top_k]
    ]


def build_messages(question: str, context: str) -> list[dict]:
    """
    Build the chat messages shared by all LLM providers.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Knowledge-base context:\n{context}\n\n"
                f"Customer question: {question}\n\n"
                "Give a clear and concise answer."
            )
        }
    ]


def generate_with_ollama(messages: list[dict]) -> str:
    """
    Send messages to the local Ollama LLM and return its response.
    """

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result.get("message", {}).get("content", "").strip()


def generate_with_groq(messages: list[dict]) -> str:
    """
    Send messages to the hosted Groq API (OpenAI-compatible)
    and return its response.
    """

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0
    }

    response = requests.post(
        GROQ_API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    choices = result.get("choices") or [{}]

    return choices[0].get("message", {}).get("content", "").strip()


def generate_answer(question: str, context: str) -> str:
    """
    Send the retrieved context and question to the configured
    LLM provider and return its response.
    """

    messages = build_messages(question, context)

    if LLM_PROVIDER == "groq":
        answer = generate_with_groq(messages)
    else:
        answer = generate_with_ollama(messages)

    if not answer:
        raise RuntimeError("The LLM returned an empty response.")

    return answer


def answer_question(question: str) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents.
    2. Check whether useful context exists.
    3. Construct the prompt.
    4. Generate an answer using Ollama.
    """

    retrieved_documents = retrieve_documents(question)

    if not retrieved_documents:
        return {
            "answer": (
                "Insufficient information in the knowledge base "
                "to answer this question."
            ),
            "sources": []
        }

    context = "\n\n".join(
        f"Document: {document['title']}\n"
        f"Content: {document['content']}"
        for document in retrieved_documents
    )

    answer = generate_answer(
        question=question,
        context=context
    )

    logger.info(
        "RAG answer generated successfully; sources=%d",
        len(retrieved_documents)
    )

    return {
        "answer": answer,
        "sources": [
            document["title"]
            for document in retrieved_documents
        ]
    }