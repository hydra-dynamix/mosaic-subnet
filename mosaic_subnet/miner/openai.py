import base64
from io import BytesIO
import httpx
from PIL import Image
from loguru import logger

from mosaic_subnet.base import BaseMiner
from mosaic_subnet.miner._config import MinerSettings


class OpenAIMiner(BaseMiner):
    """OpenAI-style miner that supports custom API endpoints."""

    def __init__(self, settings: MinerSettings) -> None:
        super().__init__()
        self.settings = settings

        if not settings.api_url:
            raise ValueError("API URL is required for OpenAI miner")
        if not settings.api_key:
            raise ValueError("API key is required for OpenAI miner")

        # Setup HTTP client
        self.client = httpx.Client(
            base_url=settings.api_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=30.0  # 30 second timeout
        )

    def generate(self, prompt: str) -> bytes:
        """Generate an image using the custom API endpoint."""
        try:
            # Make API request
            response = self.client.post(
                "/images/generations",  # Standard OpenAI-style endpoint
                json={
                    "prompt": prompt,
                    "n": 1,
                    "response_format": "b64_json"
                }
            )
            response.raise_for_status()
            result = response.json()

            # Handle standard OpenAI-style response
            if "data" in result and len(result["data"]) > 0:
                if "b64_json" in result["data"][0]:
                    return base64.b64decode(result["data"][0]["b64_json"])
                elif "url" in result["data"][0]:
                    # Download image from URL
                    img_response = httpx.get(result["data"][0]["url"])
                    img_response.raise_for_status()
                    return img_response.content

            raise ValueError(f"Unexpected API response format: {result}")

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

    def get_metadata(self) -> dict:
        return {
            "type": "openai",
            "api_url": self.settings.api_url,
            "requirements": {
                "min_ram": "2GB",  # Very light client
                "min_vram": "0GB",  # No GPU needed
                "gpu_optional": False,  # No GPU used
                "inference_type": "remote",  # Remote inference
                "avg_inference_time": "~2s"  # Network latency
            }
        }
