"""Chat model and embeddings. Uses Gemini, or offline fakes when MOCK_LLM is on."""
import json
import re

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import config


def get_embeddings():
    if config.MOCK_LLM:
        return DeterministicFakeEmbedding(size=256)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=config.GOOGLE_API_KEY)


def _mock_chat(messages):
    """Deterministic stand-in for the LLM so the pipeline can be tested offline."""
    prompt = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    if "Return ONLY a JSON object" in prompt:
        # First review fails, later reviews pass, so tests exercise the revise loop.
        passed = "REVISED DRAFT" in prompt
        return AIMessage(content=json.dumps({
            "score": 9 if passed else 5,
            "issues": [] if passed else ["Uses an exclamation mark, which the guidelines ban."],
            "suggestions": [] if passed else ["Remove exclamation marks and use the approved tagline."],
        }))
    if "Revise the draft" in prompt:
        return AIMessage(content="REVISED DRAFT: Calm, clear copy that follows the brand guidelines.")
    return AIMessage(content="FIRST DRAFT: Exciting new launch!")


def get_chat_model():
    if config.MOCK_LLM:
        return RunnableLambda(_mock_chat)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0.6)


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (models sometimes wrap it in ```json fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object in model reply")
    return json.loads(match.group(0))
