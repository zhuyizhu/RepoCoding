from functools import lru_cache
from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://repocoding:repocoding@postgres:5432/repocoding"
    redis_url: str = "redis://redis:6379/0"
    repository_root: str = "/workspace/repos"
    repograph_root: str = "/workspace/repographs"
    codelens_home: str = "/opt/codelens"
    evistate_home: str = "/opt/evistate"
    worktree_root: str = "/workspace/worktrees"
    vectorstore_root: str = "/workspace/vectorstores"
    repair_enabled: bool = False
    full_pipeline_enabled: bool = True
    repair_max_llm_calls: int = Field(default=6, ge=1, le=8)
    artifact_root: str = "/workspace/artifacts"
    sandbox_url: str = "http://java-sandbox:8080"
    cors_origins: str = "http://localhost:5173"
    event_poll_seconds: float = 0.5
    auto_create_schema: bool = False

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
