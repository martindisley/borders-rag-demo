from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import OPENAI_CHAT_URL, QueryResponse, RAGService, SourceResult, create_app, post_json


class FakeRAGService:
    def __init__(self) -> None:
        self.embedding_model = "test-embedding"
        self.chat_model = "test-chat"
        self.openai_api_key = ""
        self.chunks = [object(), object()]
        self.startup_calls: list[str] = []
        self.last_query: tuple[str, int, str | None, str | None] | None = None

    def validate_config(self) -> None:
        self.startup_calls.append("validate_config")

    def load_index(self) -> None:
        self.startup_calls.append("load_index")

    def has_server_api_key(self) -> bool:
        return bool(self.openai_api_key)

    def query(
        self,
        question: str,
        top_k: int,
        api_key_override: str | None = None,
        chat_model_override: str | None = None,
    ) -> QueryResponse:
        self.last_query = (question, top_k, api_key_override, chat_model_override)
        return QueryResponse(
            answer="Test answer [S1]",
            sources=[
                SourceResult(
                    source_id="S1",
                    chunk_id="ldp-p001-c01",
                    doc_id="ldp",
                    doc_title="Local Development Plan",
                    source_file="local-development-plan.pdf",
                    page_start=1,
                    page_end=1,
                    score=0.987654,
                    excerpt="Example excerpt",
                    source_url="https://example.test/doc#page=1",
                )
            ],
        )


class ErrorRAGService(FakeRAGService):
    def query(
        self,
        question: str,
        top_k: int,
        api_key_override: str | None = None,
        chat_model_override: str | None = None,
    ) -> QueryResponse:
        raise RuntimeError("upstream error")


def build_client(service: RAGService | FakeRAGService) -> TestClient:
    app = create_app(service)
    return TestClient(app)


def test_health_returns_service_state() -> None:
    service = FakeRAGService()

    with build_client(service) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "chunks_loaded": 2,
        "embedding_model": "test-embedding",
        "chat_model": "test-chat",
        "server_key_available": False,
    }
    assert service.startup_calls == ["validate_config", "load_index"]


def test_query_returns_structured_response() -> None:
    service = FakeRAGService()

    with build_client(service) as client:
        response = client.post(
            "/api/query",
            json={"question": "What does Policy HD1 cover?", "top_k": 3},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer [S1]"
    assert data["sources"][0]["source_id"] == "S1"
    assert service.last_query == ("What does Policy HD1 cover?", 3, None, None)


def test_query_forwards_byok_header() -> None:
    service = FakeRAGService()

    with build_client(service) as client:
        response = client.post(
            "/api/query",
            json={"question": "What does Policy HD1 cover?", "top_k": 3},
            headers={"X-OpenAI-API-Key": "sk-test-user-key"},
        )

    assert response.status_code == 200
    assert service.last_query == (
        "What does Policy HD1 cover?",
        3,
        "sk-test-user-key",
        None,
    )


def test_query_forwards_chat_model_header() -> None:
    service = FakeRAGService()

    with build_client(service) as client:
        response = client.post(
            "/api/query",
            json={"question": "What does Policy HD1 cover?", "top_k": 3},
            headers={"X-OpenAI-Chat-Model": "gpt-4o-mini"},
        )

    assert response.status_code == 200
    assert service.last_query == (
        "What does Policy HD1 cover?",
        3,
        None,
        "gpt-4o-mini",
    )


def test_query_rejects_short_question() -> None:
    with build_client(FakeRAGService()) as client:
        response = client.post("/api/query", json={"question": "hi", "top_k": 5})

    assert response.status_code == 422


def test_query_rejects_top_k_out_of_range() -> None:
    with build_client(FakeRAGService()) as client:
        response = client.post(
            "/api/query",
            json={"question": "Valid question text", "top_k": 11},
        )

    assert response.status_code == 422


def test_query_maps_runtime_error_to_http_500() -> None:
    with build_client(ErrorRAGService()) as client:
        response = client.post(
            "/api/query",
            json={"question": "Valid question text", "top_k": 2},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "upstream error"}


def test_query_rejects_unsupported_chat_model() -> None:
    service = RAGService(Path(__file__).resolve().parents[1])

    try:
        service.resolve_chat_model("o4-mini")
        assert False, "Expected resolve_chat_model to fail for unsupported model"
    except RuntimeError as exc:
        assert "Unsupported chat model" in str(exc)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
)
def test_query_accepts_supported_chat_models(model: str) -> None:
    service = RAGService(Path(__file__).resolve().parents[1])
    assert service.resolve_chat_model(model) == model


@pytest.mark.integration
def test_openai_chat_completion_integration() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("INTEGRATION_OPENAI_MODEL", "gpt-5-nano")
    response = post_json(
        OPENAI_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload={
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with exactly: ok"},
            ],
        },
    )

    assert "choices" in response
    assert len(response["choices"]) > 0
