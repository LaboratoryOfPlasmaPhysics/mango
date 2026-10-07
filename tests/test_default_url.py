from space_mango.client import DEFAULT_URL, MangoClient


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "http://127.0.0.1:8765/")
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == "http://127.0.0.1:8765"


def test_unset_env_falls_back_to_public_server(monkeypatch, tmp_path):
    monkeypatch.delenv("SPACE_MANGO_URL", raising=False)
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == DEFAULT_URL.rstrip("/")


def test_empty_env_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "")
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == DEFAULT_URL.rstrip("/")


def test_explicit_base_url_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "http://ignored:1")
    c = MangoClient("http://explicit:2", cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == "http://explicit:2"
