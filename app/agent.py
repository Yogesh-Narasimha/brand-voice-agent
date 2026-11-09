"""LangGraph agent: retrieve guidelines -> draft -> review -> revise (loop) -> final content.

    retrieve -> draft -> review --(score >= PASS_SCORE or out of revisions)--> END
                           ^            |
                           +-- revise <-+
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app import config
from app.brand_kits import retrieve
from app.llm import get_chat_model, parse_json

CHANNEL_HINTS = {
    "email": "a marketing email with a subject line and a short body (under 150 words)",
    "linkedin": "a LinkedIn post (under 120 words, professional, at most 3 hashtags)",
    "instagram": "an Instagram caption (under 60 words, at most 5 hashtags)",
    "ad": "a search ad with a headline (max 30 characters) and description (max 90 characters)",
    "product": "a product description (under 100 words)",
}

SYSTEM = SystemMessage(content=(
    "You are a senior brand copywriter. You write only content that follows the brand guidelines "
    "you are given. If the guidelines and the brief conflict, the guidelines win."
))


class AgentState(TypedDict, total=False):
    kit_id: str
    brief: str
    channel: str
    guidelines: list[str]
    sources: list[dict]
    draft: str
    review: dict
    revisions: int
    history: list[dict]


def _guideline_block(state: AgentState) -> str:
    return "\n\n".join(f"[{i + 1}] {g}" for i, g in enumerate(state["guidelines"]))


def retrieve_node(state: AgentState) -> AgentState:
    docs = retrieve(state["kit_id"], f"{state['channel']} content: {state['brief']}")
    return {
        "guidelines": [d.page_content for d in docs],
        "sources": [{"file": d.metadata.get("source"), "chunk": d.metadata.get("chunk")} for d in docs],
        "revisions": 0,
        "history": [],
    }


def draft_node(state: AgentState) -> AgentState:
    prompt = (
        f"Brand guidelines:\n{_guideline_block(state)}\n\n"
        f"Brief: {state['brief']}\n\n"
        f"Write {CHANNEL_HINTS[state['channel']]}. Return only the content."
    )
    draft = get_chat_model().invoke([SYSTEM, HumanMessage(content=prompt)]).content
    return {"draft": draft, "history": state["history"] + [{"step": "draft", "content": draft}]}


def review_node(state: AgentState) -> AgentState:
    prompt = (
        f"Brand guidelines:\n{_guideline_block(state)}\n\n"
        f"Content to review:\n{state['draft']}\n\n"
        "Check the content strictly against the guidelines (tone, banned words, formatting, claims, "
        "required terms) and the channel limits. Return ONLY a JSON object: "
        '{"score": <integer 0-10>, "issues": [<strings>], "suggestions": [<strings>]}'
    )
    reply = get_chat_model().invoke([HumanMessage(content=prompt)]).content
    try:
        review = parse_json(reply)
        review["score"] = int(review.get("score", 0))
        review.setdefault("issues", [])
        review.setdefault("suggestions", [])
    except (ValueError, TypeError):
        review = {"score": 0, "issues": ["Reviewer reply was not valid JSON."], "suggestions": []}
    return {"review": review, "history": state["history"] + [{"step": "review", **review}]}


def revise_node(state: AgentState) -> AgentState:
    review = state["review"]
    prompt = (
        f"Brand guidelines:\n{_guideline_block(state)}\n\n"
        f"Brief: {state['brief']}\n\n"
        f"Current draft:\n{state['draft']}\n\n"
        f"Reviewer issues: {review['issues']}\nReviewer suggestions: {review['suggestions']}\n\n"
        f"Revise the draft to fix every issue. It must still be {CHANNEL_HINTS[state['channel']]}. "
        "Return only the revised content."
    )
    draft = get_chat_model().invoke([SYSTEM, HumanMessage(content=prompt)]).content
    return {
        "draft": draft,
        "revisions": state["revisions"] + 1,
        "history": state["history"] + [{"step": "revise", "content": draft}],
    }


def after_review(state: AgentState) -> str:
    if state["review"]["score"] >= config.PASS_SCORE or state["revisions"] >= config.MAX_REVISIONS:
        return END
    return "revise"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("draft", draft_node)
    g.add_node("review", review_node)
    g.add_node("revise", revise_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", after_review, {"revise": "revise", END: END})
    g.add_edge("revise", "review")
    return g.compile()


graph = build_graph()


def generate(kit_id: str, brief: str, channel: str) -> dict:
    state = graph.invoke({"kit_id": kit_id, "brief": brief, "channel": channel})
    return {
        "content": state["draft"],
        "review": state["review"],
        "passed": state["review"]["score"] >= config.PASS_SCORE,
        "revisions": state["revisions"],
        "sources": state["sources"],
        "history": state["history"],
    }
