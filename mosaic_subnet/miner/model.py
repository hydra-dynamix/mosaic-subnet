from typing import Optional, Dict, Any
import threading
import base64
from io import BytesIO
import httpx
from communex.module.module import Module, endpoint

class CustomAPIImageMiner(Module):
    def __init__(self, api_key: str, base_url: str, model: str = "", 
                 additional_headers: Dict[str, str] = None,
                 additional_params: Dict[str, Any] = None) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                **(additional_headers or {})
            },
            timeout=60.0  # Adjust timeout as needed for your inference
        )
        self.additional_params = additional_params or {}
        self._lock = threading.Lock()

    @endpoint
    def sample(
        self, 
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate an image using the custom inference API endpoint.
        Returns a base64-encoded PNG image to maintain compatibility with the original interface.
        
        Args:
            prompt: The text prompt for image generation
            negative_prompt: Optional negative prompt for guidance
            steps: Number of inference steps
            seed: Optional seed for reproducibility
            **kwargs: Additional parameters that will be passed to the API
        """
        with self._lock:
            # Combine default parameters with any additional parameters
            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                **({} if seed is None else {"seed": seed}),
                **self.additional_params,
                **kwargs
            }
            
            if self.model:
                params["model"] = self.model

            # Make the API call
            response = self.client.post(
                f"{self.base_url}/generate",  # Adjust endpoint path as needed
                json=params
            )
            response.raise_for_status()
            
            # Extract the base64 image from the response
            # Adjust this based on your API's response format
            result = response.json()
            
            # The response format needs to match your API's output format
            # This is just an example - adjust according to your API
            if "image" in result:
                return result["image"]  # Assuming API returns base64 image
            elif "images" in result:
                return result["images"][0]  # Some APIs return a list
            else:
                raise ValueError(f"Unexpected API response format: {result}")

    @endpoint
    def get_metadata(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "capabilities": {
                # Add any specific capabilities of your model/endpoint here
                "supports_negative_prompt": True,
                "supports_seed": True,
            }
        }

if __name__ == "__main__":
    # Example usage
    miner = CustomAPIImageMiner(
        api_key="YOUR_API_KEY",
        base_url="http://your-inference-endpoint",
        model="your-model-name",
        additional_headers={"Custom-Header": "value"},
        additional_params={"custom_param": "value"}
    )
    out = miner.sample(
        prompt="A beautiful sunset over mountains",
        negative_prompt="blur, haze",
        steps=30
    )
    # Save the base64 image for testing
    with open("test_image.png", "wb") as f:
        f.write(base64.b64decode(out))
