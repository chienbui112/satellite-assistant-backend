"""
FastAPI dependency wiring. Services are singletons cached per process.
"""

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.external_provider_service import ExternalProviderService
from app.services.geocode_service import GeocodeService
from app.services.llm_service import LLMService
from app.services.stac_service import STACService
from app.services.tokenizer_service import Tokenizer, build_tokenizer


@lru_cache
def get_stac_service() -> STACService:
    return STACService(get_settings())


@lru_cache
def get_external_provider_service() -> ExternalProviderService:
    return ExternalProviderService(get_settings())


@lru_cache
def get_geocode_service() -> GeocodeService:
    return GeocodeService(get_settings())


@lru_cache
def get_tokenizer() -> Tokenizer:
    return build_tokenizer(get_settings())


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService(
        get_settings(),
        get_stac_service(),
        get_geocode_service(),
        get_tokenizer(),
        get_external_provider_service(),
    )


# Re-exported for routers to depend on without importing the factories directly.
SettingsDep = Depends(get_settings)
STACDep = Depends(get_stac_service)
GeocodeDep = Depends(get_geocode_service)
TokenizerDep = Depends(get_tokenizer)
LLMDep = Depends(get_llm_service)

__all__ = [
    "Settings",
    "get_settings",
    "get_stac_service",
    "get_external_provider_service",
    "get_geocode_service",
    "get_tokenizer",
    "get_llm_service",
    "SettingsDep",
    "STACDep",
    "GeocodeDep",
    "TokenizerDep",
    "LLMDep",
]
