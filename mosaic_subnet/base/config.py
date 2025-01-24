from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class MosaicBaseSettings(BaseSettings):
    use_testnet: bool = False
    call_timeout: int = 60

    # TODO: whitelist&blacklist
    # whitelist: List[str] = []
    # blacklist: List[str] = []

    class Config:
        env_prefix = "MOSAIC_"
        env_file = os.getenv("MOSAIC_ENV_FILE", "env/config.env")
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields for flexibility

    @classmethod
    def get_env_file_path(cls) -> Path:
        """Get the environment file path, creating parent directories if needed."""
        env_path = Path(cls.Config.env_file)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        return env_path
