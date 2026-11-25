"""Simple TF-IDF + Cosine Similarity RAG chatbot ('Sheti Mitra') using Gemini 2.5 Flash."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import ChatResponse, ChatSource
from services.gemini_utils import (
    OVERLOAD_FALLBACK_GEMINI_MODELS,
    PRIMARY_GEMINI_MODEL,
    build_gemini_chat_kwargs,
    extract_json_payload as _extract_json,
    extract_message_text as _extract_message_text,
    invoke_gemini_with_503_recovery,
)

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "farming.json"
DEFAULT_CHAT_MODEL = PRIMARY_GEMINI_MODEL
MIN_RELEVANCE_SCORE = 0.11

SYSTEM_PROMPT = """You are "Sheti Mitra", a helpful, practical farming assistant.
Answer the farmer's question using the 3 retrieved farming notes provided below.

Rules:
1. Reply in {target_language} using simple, clear, practical words for farmers.
2. Ground specific facts, steps, and numbers in the retrieved notes. Do NOT hallucinate or invent exact chemical dosages, yield guarantees, fake statistics, or external URLs.
3. For normal farming questions (for example, asking which crop to sow, general soil care, sowing, or irrigation advice), give a helpful, clear answer based on the retrieved notes. If choosing an exact crop depends on their soil nutrients and location weather, also suggest entering their N, P, K, pH, and location in the Crop Recommendation form above.
4. If the question is clearly unrelated to farming or cannot be answered honestly without guessing outside the notes (such as sports, politics, crypto, entertainment, or exact unsupported product dosages), set "covered" to false and clearly say in {target_language} that the current farming notes do not cover that question.
5. When covered is true, cite which note IDs (from the 3 retrieved notes) you used in "cited_note_ids".
6. Return valid JSON only with keys: "covered" (boolean), "answer" (string), and "cited_note_ids" (list of strings)."""


class GeminiChatPayload(BaseModel):
    covered: bool
    answer: str = Field(..., min_length=3)
    cited_note_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class FarmingNote:
    id: str
    topic: str
    title_en: str
    title_mr: str
    content_en: str
    content_mr: str
    keywords: str
    source_name: str
    source_url: str

    def to_searchable_text(self) -> str:
        return " ".join(
            [
                self.topic.replace("_", " "),
                self.title_en,
                self.title_mr,
                self.content_en,
                self.content_mr,
                self.keywords,
            ]
        )


class ShetiMitraRAG:
    """In-memory TF-IDF + Cosine Similarity retriever over data/farming.json."""

    def __init__(self, notes: list[FarmingNote]) -> None:
        self.notes = notes
        # Combine Unicode-safe word n-grams and char_wb n-grams for English & Marathi.
        self.word_vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s\.,?!।:;\"'()\[\]{}/-]+",
            ngram_range=(1, 2),
        )
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        corpus = [note.to_searchable_text() for note in notes]
        if corpus:
            self.word_matrix = self.word_vectorizer.fit_transform(corpus)
            self.char_matrix = self.char_vectorizer.fit_transform(corpus)
        else:
            self.word_matrix = None
            self.char_matrix = None

    @classmethod
    def from_json(cls, path: Path = DEFAULT_KB_PATH) -> "ShetiMitraRAG":
        with open(path, "r", encoding="utf-8") as file:
            raw_notes = json.load(file)
        notes = [FarmingNote(**item) for item in raw_notes]
        return cls(notes)

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[FarmingNote, float]]:
        if not self.notes or self.word_matrix is None or self.char_matrix is None:
            return []
        cleaned = question.strip()
        if not cleaned:
            return []
        word_sims = cosine_similarity(
            self.word_vectorizer.transform([cleaned]), self.word_matrix
        )[0]
        char_sims = cosine_similarity(
            self.char_vectorizer.transform([cleaned]), self.char_matrix
        )[0]
        similarities = 0.5 * word_sims + 0.5 * char_sims
        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda idx: float(similarities[idx]),
            reverse=True,
        )[:top_k]
        return [(self.notes[idx], float(similarities[idx])) for idx in ranked_indices]


def _build_gemini_chat_model(api_key: str, model_name: str = DEFAULT_CHAT_MODEL) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    timeout = 14.0 if model_name != PRIMARY_GEMINI_MODEL else 15.0
    kwargs = build_gemini_chat_kwargs(
        model_name=model_name,
        api_key=api_key,
        timeout=timeout,
        temperature=0.2,
    )
    return ChatGoogleGenerativeAI(**kwargs)


def _not_covered_message(language: Literal["en", "mr"]) -> str:
    if language == "mr":
        return (
            "माफ करा, माझ्या सध्याच्या शेती माहिती संकलनामध्ये (माती परीक्षण, सिंचन, पेरणी आणि पीक काळजी) "
            "या प्रश्नाची माहिती उपलब्ध नाही. कृपया या विषयांशी संबंधित प्रश्न विचारा."
        )
    return (
        "Sorry, my current farming notes (soil testing, irrigation, sowing, and crop care) "
        "do not cover that question. Please ask a question related to those farming topics."
    )


def _unavailable_message(
    language: Literal["en", "mr"],
    top_note: FarmingNote | None = None,
    reason: str | None = None,
) -> str:
    if top_note is not None:
        if language == "mr":
            return (
                f"{top_note.title_mr}: {top_note.content_mr}\n\n"
                "(AI सेवा सध्या उपलब्ध नाही. संबंधित शेती नोंद दाखवली आहे.)"
            )
        return (
            f"{top_note.title_en}: {top_note.content_en}\n\n"
            "(AI service is temporarily unavailable. Showing the relevant farming note.)"
        )

    if language == "mr":
        return (
            "सध्या शेती मित्र AI सेवा उपलब्ध नाही (सर्व्हरवर GEMINI_API_KEY कॉन्फिगर केलेली नाही किंवा "
            "कनेक्शनमध्ये अडचण आहे). खाली दिलेल्या संबंधित नोंदी व अधिकृत दुवे पहा."
        )
    detail = f" ({reason})" if reason else ""
    return (
        f"Sheti Mitra AI service is currently unavailable{detail}. "
        "Please check GEMINI_API_KEY in .env or review the retrieved official notes below."
    )


def answer_farming_question(
    rag: ShetiMitraRAG,
    question: str,
    language: Literal["en", "mr"] = "en",
    chat_model: BaseChatModel | None = None,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> ChatResponse:
    """Retrieve top 3 farming notes via TF-IDF + cosine similarity and answer via Gemini 2.5 Flash."""
    retrieved = rag.retrieve(question, top_k=3)
    top_score = retrieved[0][1] if retrieved else 0.0

    if not retrieved or top_score < min_score:
        return ChatResponse(
            question=question.strip(),
            language=language,
            answer=_not_covered_message(language),
            covered=False,
            sources=[],
            status="not_covered",
            message="No sufficiently relevant note found in data/farming.json.",
        )

    notes_by_id = {note.id: (note, score) for note, score in retrieved}
    all_sources = [
        ChatSource(
            id=note.id,
            title=note.title_mr if language == "mr" else note.title_en,
            topic=note.topic,
            source_name=note.source_name,
            url=note.source_url,
            score=round(score, 4),
        )
        for note, score in retrieved
    ]

    custom_chat_model = chat_model is not None
    model_name = DEFAULT_CHAT_MODEL
    api_key = ""
    if chat_model is None:
        api_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        if not api_key:
            return ChatResponse(
                question=question.strip(),
                language=language,
                answer=_unavailable_message(language, retrieved[0][0], "GEMINI_API_KEY missing"),
                covered=True,
                sources=all_sources,
                status="unavailable",
                message="GEMINI_API_KEY is not configured in .env.",
            )
        model_name = os.getenv("CHAT_LLM_MODEL", "").strip() or DEFAULT_CHAT_MODEL
        chat_model = _build_gemini_chat_model(api_key=api_key, model_name=model_name)

    target_lang_name = "Marathi (मराठी)" if language == "mr" else "English"
    notes_context = "\n\n".join(
        (
            f"[{note.id}] ({note.topic})\n"
            f"English Title: {note.title_en}\n"
            f"English Note: {note.content_en}\n"
            f"Marathi Title: {note.title_mr}\n"
            f"Marathi Note: {note.content_mr}\n"
            f"Source: {note.source_name} ({note.source_url})"
        )
        for note, _ in retrieved
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(target_language=target_lang_name)),
        HumanMessage(
            content=(
                f"Farmer Question ({target_lang_name}): {question.strip()}\n\n"
                f"Retrieved Notes:\n{notes_context}"
            )
        ),
    ]

    allow_overload_fallback = (
        not custom_chat_model
        and bool(api_key)
        and model_name == PRIMARY_GEMINI_MODEL
    )

    def _invoke_chat(override_model: str | None) -> Any:
        active_model = chat_model
        if override_model is not None:
            active_model = _build_gemini_chat_model(api_key=api_key, model_name=override_model)
        return active_model.invoke(messages)

    used_model = model_name
    try:
        ai_msg, used_model = invoke_gemini_with_503_recovery(
            _invoke_chat,
            is_gemini=True,
            primary_model=model_name,
            allow_model_fallback=allow_overload_fallback,
            fallback_models=OVERLOAD_FALLBACK_GEMINI_MODELS,
            sleep_fn=time.sleep,
            logger=logger,
            context_label="Sheti Mitra",
        )
        raw_content = _extract_message_text(ai_msg)
        parsed = GeminiChatPayload.model_validate_json(_extract_json(raw_content))
    except Exception as exc:
        logger.warning("Sheti Mitra %s failed (%s)", used_model, type(exc).__name__)
        return ChatResponse(
            question=question.strip(),
            language=language,
            answer=_unavailable_message(language, retrieved[0][0]),
            covered=True,
            sources=all_sources,
            status="unavailable",
            message=f"Gemini chat call failed ({type(exc).__name__}).",
        )

    if not parsed.covered:
        return ChatResponse(
            question=question.strip(),
            language=language,
            answer=parsed.answer.strip() or _not_covered_message(language),
            covered=False,
            sources=[],
            status="not_covered",
            message=None,
        )

    valid_cited = [
        ChatSource(
            id=notes_by_id[nid][0].id,
            title=(
                notes_by_id[nid][0].title_mr
                if language == "mr"
                else notes_by_id[nid][0].title_en
            ),
            topic=notes_by_id[nid][0].topic,
            source_name=notes_by_id[nid][0].source_name,
            url=notes_by_id[nid][0].source_url,
            score=round(notes_by_id[nid][1], 4),
        )
        for nid in dict.fromkeys(parsed.cited_note_ids)
        if nid in notes_by_id
    ]
    if not valid_cited:
        valid_cited = all_sources

    return ChatResponse(
        question=question.strip(),
        language=language,
        answer=parsed.answer.strip(),
        covered=True,
        sources=valid_cited,
        status="ok",
        message=None,
    )
