from enum import Enum
from functools import lru_cache
from typing import Annotated, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"


class TokenizerMethod(str, Enum):
    TRANSFORMERS = "transformers"  # exact, large dep
    TIKTOKEN = "tiktoken"           # close-enough, lightweight
    CHAR_COUNT = "char_count"       # zero-dep fallback (len(text)//2)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider switch ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_temperature: float = 0.1

    # --- Ollama (local) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_num_ctx: int = 8192

    # --- Anthropic ---
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6"

    # --- OpenAI (also covers OpenAI-compatible endpoints via base_url) ---
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None  # set for Azure / proxies / OpenRouter

    # --- Google Gemini ---
    google_api_key: Optional[str] = None
    google_model: str = "gemini-2.0-flash"

    # --- Groq ---
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"

    # --- STAC ---
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_default_limit: int = 10
    stac_max_limit: int = 50

    # --- External commercial-provider aggregator ---
    # POST endpoint that fronts Maxar / Planet / AxelGlobe.
    # Defaults to the geohub.vn relay; set to empty string to force mock mode.
    external_provider_url: str = "https://api.geohub.vn/v1/vmrs/api/v1/search"
    external_provider_api_key: Optional[str] = None
    # Raw cookie string sent as the `Cookie` header on every aggregator call,
    # e.g. "session=abc123; csrftoken=xyz". Required when the upstream rejects
    # bearer-token-only requests.
    external_provider_cookie: Optional[str] = None
    # Default page size sent to the external API. Mapped from frontend
    # `limit`; this is the fallback when the request omits it.
    external_provider_page_size: int = 10

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # NoDecode tells pydantic-settings to skip its built-in JSON parsing for
    # this field, so our `split_cors` validator below receives the raw string
    # (e.g. "http://a, http://b") instead of having JSON-decode fail on it.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    log_level: str = "INFO"

    # --- Conversation history window ---
    # How many *past* user-anchored turns of history to send to the LLM each
    # turn. A "turn" is one user message + the assistant reply (+ any tool
    # messages in between). Set to a positive integer to limit; leave empty
    # or set to "none"/"null" to send the full history (unlimited). Pydantic
    # can't parse "" / "none" as Optional[int] without help, so the validator
    # below normalises the env-string into None or int.
    chat_history_window_size: Optional[int] = None

    # --- Tokenizer (for the token usage progress bar) ---
    tokenizer_method: TokenizerMethod = TokenizerMethod.TIKTOKEN
    # Only used when tokenizer_method == "transformers".
    tokenizer_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # Budget the UI shows. Default 6000 = Groq qwen3-32b free tier TPM ceiling.
    max_tokens: int = 6000
    # Above this, the progress bar turns red.
    warning_threshold: int = 4500

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("chat_history_window_size", mode="before")
    @classmethod
    def parse_history_window(cls, v):
        if v is None or isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        try:
            return int(s)
        except ValueError:
            raise ValueError(
                "CHAT_HISTORY_WINDOW_SIZE must be an integer, 'none', 'null', "
                f"or empty. Got: {v!r}"
            )

    @model_validator(mode="after")
    def check_provider_credentials(self) -> "Settings":
        """Fail fast at startup if the selected provider has no credentials.

        We deliberately don't error on the *other* providers' missing keys —
        users typically only configure the one they're actually using.
        """
        required_for = {
            LLMProvider.ANTHROPIC: ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            LLMProvider.OPENAI: ("openai_api_key", "OPENAI_API_KEY"),
            LLMProvider.GOOGLE: ("google_api_key", "GOOGLE_API_KEY"),
            LLMProvider.GROQ: ("groq_api_key", "GROQ_API_KEY"),
        }
        pair = required_for.get(self.llm_provider)
        if pair:
            attr, env_name = pair
            if not getattr(self, attr):
                raise ValueError(
                    f"LLM_PROVIDER={self.llm_provider.value} requires {env_name} "
                    f"to be set in the environment or .env file"
                )
        return self

    # Convenience: the *active* model name for logging / /healthz.
    @property
    def active_model(self) -> str:
        return {
            LLMProvider.OLLAMA: self.ollama_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.GOOGLE: self.google_model,
            LLMProvider.GROQ: self.groq_model,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
