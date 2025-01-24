from dataclasses import dataclass
from typing import Literal, Optional

from mosaic_subnet.base.config import BaseSettings


@dataclass
class MinerSettings(BaseSettings):
    """Settings for the Mosaic miner."""
    
    # Provider settings
    provider: Literal["openai", "local"] = "openai"  # openai for custom endpoints, local for local inference
    
    # Local provider settings
    model: str = "stabilityai/sdxl-turbo"  # Only used when provider=local
    
    # Custom endpoint settings (when provider=openai)
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Network settings
    host: str = "0.0.0.0"
    port: int = 8080
