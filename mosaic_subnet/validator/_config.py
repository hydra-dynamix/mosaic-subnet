from dataclasses import dataclass
from typing import Literal, Optional

from mosaic_subnet.base.config import BaseSettings


@dataclass
class ValidatorSettings(BaseSettings):
    """Settings for the Mosaic validator."""
    
    # Model settings
    model: Literal["hps", "clip", "api"] = "hps"  # hps/clip for local, api for remote
    
    # CLIP model settings (when model=clip)
    clip_model: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    
    # API settings (when model=api)
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Network settings
    host: str = "0.0.0.0"
    port: int = 8081
    
    # Timing settings
    iteration_interval: int = 60  # seconds
    call_timeout: int = 30  # seconds
