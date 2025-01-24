from io import BytesIO
import base64
import httpx
from PIL import Image
from communex.module.module import Module
from loguru import logger


class APIValidator(Module):
    """API-based validator that uses a remote endpoint for scoring."""
    
    def __init__(self, api_url: str, api_key: str = None) -> None:
        super().__init__()
        self.api_url = api_url.rstrip('/')
        
        # Setup HTTP client
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        self.client = httpx.Client(
            headers=headers,
            timeout=30.0  # 30 second timeout
        )

    def get_similarity(self, file: bytes, prompt: str) -> float:
        """Get similarity score using remote API."""
        # Convert image to base64
        image = Image.open(BytesIO(file))
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Make API request
        response = self.client.post(
            f"{self.api_url}/score",
            json={
                "image": image_b64,
                "prompt": prompt
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract score from response
        if "score" in result:
            return float(result["score"])
        elif "similarity" in result:
            return float(result["similarity"])
        else:
            raise ValueError(f"Unexpected API response format: {result}")

    def get_metadata(self) -> dict:
        return {
            "type": "api",
            "api_url": self.api_url,
            "requirements": {
                "min_ram": "2GB",  # Very light client
                "min_vram": "0GB",  # No GPU needed
                "gpu_optional": False,  # No GPU used
                "inference_type": "remote_scoring",
                "avg_inference_time": "~200ms"  # Network latency adds some time
            }
        }
