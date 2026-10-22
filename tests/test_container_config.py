from pathlib import Path

import yaml

from serviceflow.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_database_connection_can_be_overridden_for_container():
    settings = Settings(_env_file=None, deepseek_api_key="test",
                        checkpoint_database_url="postgresql://agent:password@postgres/db",
                        database_url="postgresql+psycopg://agent:password@postgres/db")
    assert "@postgres/" in settings.database_url
    assert settings.database_echo is False


def test_compose_preserves_database_volume_and_opt_in_api():
    config = yaml.safe_load((ROOT / "compose.yaml").read_text())
    assert "agent-study-postgres-data:/var/lib/postgresql/data" in config["services"]["postgres"]["volumes"]
    api = config["services"]["api"]
    assert api["profiles"] == ["app"]
    assert "127.0.0.1:8001:8000" in api["ports"]
    for name in ("DATABASE_URL", "CHECKPOINT_DATABASE_URL"):
        assert "@postgres:5432/" in api["environment"][name]
    assert api["depends_on"]["postgres"]["condition"] == "service_healthy"


def test_dockerfile_does_not_copy_secret_or_local_venv():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY . ." not in dockerfile
    assert "COPY .env" not in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in ignored
    assert ".venv" in ignored


def test_fresh_compose_isolates_database_and_api_port():
    config = yaml.safe_load((ROOT / "compose.fresh.yaml").read_text())
    db = config["services"]["postgres"]
    assert "ports" not in db
    assert db["volumes"] == ["fresh-postgres-data:/var/lib/postgresql/data"]
    assert "container_name" not in db
    assert config["services"]["api"]["ports"] == ["127.0.0.1:8002:8000"]
    assert all("name" not in value and not value.get("external") for value in config["volumes"].values() if value)
