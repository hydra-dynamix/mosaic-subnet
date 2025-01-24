from typing import Optional
import base64
from io import BytesIO
import threading
import torch
from diffusers import AutoPipelineForText2Image
from communex.module.module import Module, endpoint
from communex.client import CommuneClient
from communex.types import Ss58Address
from substrateinterface import Keypair

from mosaic_subnet.base.utils import get_netuid
from mosaic_subnet.miner._config import MinerSettings

class LocalMiner(Module):
    """Local miner that uses diffusers for inference."""
    
    def __init__(self, key: Keypair, settings: MinerSettings = None) -> None:
        super().__init__()
        self.settings = settings or MinerSettings()
        
        # Setup local pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.settings.api_model or "stabilityai/sdxl-turbo"
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            variant="fp16" if self.device.type == "cuda" else None
        ).to(self.device)
        
        # Setup commune client
        self.key = key
        self.c_client = CommuneClient(
            get_node_url(use_testnet=self.settings.use_testnet)
        )
        self.netuid = get_netuid(self.c_client)
        self._lock = threading.Lock()

    @endpoint
    def sample(
        self,
        prompt: str,
        steps: int = 50,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate an image using the local pipeline.
        Returns a base64-encoded image string.
        """
        generator = torch.Generator(self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        with self._lock:
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=0.0
            ).images[0]
            
            buf = BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

    @endpoint
    def get_metadata(self) -> dict:
        """Get information about the miner and its capabilities."""
        return {
            "type": "local",
            "model": self.settings.api_model or "stabilityai/sdxl-turbo",
            "device": str(self.device),
            "capabilities": {
                "supports_negative_prompt": True,
                "supports_seed": True,
            }
        }

    def serve(self):
        """Start serving the miner on the specified host and port."""
        from communex.module.server import ModuleServer
        import uvicorn

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()
        uvicorn.run(app, host=self.settings.host, port=self.settings.port)

if __name__ == "__main__":
    from mosaic_subnet.miner._config import MinerSettings
    from substrateinterface import Keypair

    settings = MinerSettings()
    key = Keypair.create_from_uri(settings.key_uri)
    miner = LocalMiner(key, settings)
    miner.serve()
