
from src.config import settings


def test_required_settings_have_valid_defaults():
    assert settings.app_name
    assert settings.app_env in {"development", "testing", "production"}
    assert settings.chunk_size > 0
    assert 0 <= settings.chunk_overlap < settings.chunk_size
    assert settings.retrieval_top_k > 0
    assert settings.rerank_top_k > 0


def test_rerank_count_does_not_exceed_retrieval_count():
    assert settings.rerank_top_k <= settings.retrieval_top_k
