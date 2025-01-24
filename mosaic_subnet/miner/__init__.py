from mosaic_subnet.miner.openai import OpenAIMiner
from mosaic_subnet.miner.local import LocalMiner

# Default miner is OpenAIMiner for custom endpoints
Miner = OpenAIMiner

__all__ = [
    'Miner',
    'OpenAIMiner',
    'LocalMiner'
]

from communex.module.module import Module
from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex.compat.key import check_ss58_address
from communex.types import Ss58Address
from substrateinterface import Keypair
from communex.key import generate_keypair
from communex.compat.key import classic_load_key
from communex._common import get_node_url

from mosaic_subnet.miner.model import CustomAPIImageMiner
from mosaic_subnet.miner._config import MinerSettings
from mosaic_subnet.base.utils import get_netuid
import sys

from loguru import logger


class Miner(CustomAPIImageMiner):
    def __init__(self, key: Keypair, settings: MinerSettings = None) -> None:
        self.settings = settings or MinerSettings()
        super().__init__(
            api_key=self.settings.api_key,
            base_url=self.settings.api_base_url,
            model=self.settings.api_model,
            additional_headers=self.settings.api_additional_headers,
            additional_params=self.settings.api_additional_params
        )
        self.key = key
        self.c_client = CommuneClient(
            get_node_url(use_testnet=self.settings.use_testnet)
        )
        self.netuid = get_netuid(self.c_client)

    def serve(self):
        from communex.module.server import ModuleServer
        import uvicorn

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()
        uvicorn.run(app, host=self.settings.host, port=self.settings.port)


if __name__ == "__main__":
    settings = MinerSettings(
        host="0.0.0.0",
        port=7777,
        use_testnet=True,
        api_key="YOUR_API_KEY",
        api_base_url="http://your-inference-endpoint",
        api_model="your-model-name",
        api_additional_headers={"Custom-Header": "value"},
        api_additional_params={"custom_param": "value"}
    )
    Miner(key=classic_load_key("mosaic-miner0"), settings=settings).serve()
