from __future__ import annotations

from fastapi.testclient import TestClient

from app import QueryResponse, RAGService, SourceResult, create_app


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
