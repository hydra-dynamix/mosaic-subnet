import asyncio
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import List

from communex._common import get_node_url
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from communex.module.module import Module, endpoint
from loguru import logger
from pydantic import BaseModel
from substrateinterface import Keypair

from mosaic_subnet.base import SampleInput, BaseValidator
from mosaic_subnet.base.utils import get_netuid
from mosaic_subnet.validator._config import ValidatorSettings
from mosaic_subnet.validator.dataset import ValidationDataset
from mosaic_subnet.validator.model import HPS, CLIP
from mosaic_subnet.validator.api import APIValidator
from mosaic_subnet.validator.utils import normalize_score, weight_score


class WeightHistory(BaseModel):
    time: datetime
    data: List


class Validator(BaseValidator, Module):
    def __init__(self, key: Keypair, settings: ValidatorSettings | None = None) -> None:
        super().__init__()
        super(BaseValidator, self).__init__()
        self.settings = settings or ValidatorSettings()
        self.key = key

        self.netuid = get_netuid(self.c_client)
        
        # Initialize model based on settings
        if self.settings.model == "clip":
            self.model = CLIP(model_name=self.settings.clip_model)
        elif self.settings.model == "api":
            if not self.settings.api_url:
                raise ValueError("API URL is required when using api model")
            self.model = APIValidator(
                api_url=self.settings.api_url,
                api_key=self.settings.api_key
            )
        else:
            self.model = HPS()
            
        self.dataset = ValidationDataset()
        self.call_timeout = self.settings.call_timeout
        self.weights_histories = deque(maxlen=10)

    @property
    def c_client(self):
        return CommuneClient(get_node_url(use_testnet=self.settings.use_testnet))

    def calculate_score(self, img: bytes, prompt: str):
        try:
            return self.model.get_similarity(img, prompt)
        except Exception as e:
            logger.error(e)
            return 0.0

    async def validate_step(self):
        score_dict = dict()
        duration_dict = dict()
        modules_info = self.get_queryable_miners()

        input = self.get_validate_input()
        logger.debug("input: {}", input)
        futures = []
        for miner_info in modules_info.values():
            future = self.get_miner_generation_with_elapsed(miner_info, input)
            futures.append(future)
        miner_answers = await asyncio.gather(*futures)
        for uid, miner_response in zip(modules_info.keys(), miner_answers):
            miner_answer, elapsed = miner_response
            if not miner_answer:
                logger.debug(f"Skipping miner {uid}: no answer")
                continue
            score = self.calculate_score(miner_answer, input.prompt)
            if score == 0:
                logger.debug(f"Skipping miner {uid}: score is 0")
                continue
            logger.debug(f"uid {uid}, score: {score}, elapsed time: {elapsed}")
            score_dict[uid] = score
            duration_dict[uid] = elapsed

        if not score_dict:
            logger.info("score_dict empty, skip set weights")
            return

        normalized_scores = normalize_score(score_dict, duration_dict)
        weighted_scores = weight_score(normalized_scores)

        weight_data = list(
            zip(
                weighted_scores.keys(),
                score_dict.values(),
                duration_dict.values(),
                normalized_scores.values(),
                weighted_scores.values(),
            )
        )
        logger.debug("scores: {}", weight_data)
        self.weights_histories.append(
            WeightHistory(
                time=datetime.now(),
                data=weight_data,
            )
        )

        weighted_scores = {k: v for k, v in weighted_scores.items() if v > 0}
        if not weighted_scores:
            logger.info("weighted_scores empty, skip set weights")
            return
        try:
            uids = list(weighted_scores.keys())
            weights = list(weighted_scores.values())
            logger.info("Setting weights for {count} uids", count=len(uids))
            logger.debug(f"Setting weights for the following uids: {weighted_scores}")
            self.c_client.vote(
                key=self.key, uids=uids, weights=weights, netuid=self.netuid
            )
        except Exception as e:
            logger.error(e)

    def get_validate_input(self):
        return SampleInput(
            prompt=self.dataset.random_prompt(),
            steps=4,
        )

    def validation_loop(self) -> None:
        settings = self.settings
        while True:
            try:
                logger.info(f"run validation loop")
                start_time = time.time()
                asyncio.run(self.validate_step())
                elapsed = time.time() - start_time
                if elapsed < settings.iteration_interval:
                    sleep_time = settings.iteration_interval - elapsed
                    logger.info(f"Sleeping for {sleep_time}")
                    time.sleep(sleep_time)
            except Exception as e:
                print(traceback.format_exc())

    def start_validation_loop(self):
        logger.info("start sync loop")
        self._loop_thread = threading.Thread(target=self.validation_loop, daemon=True)
        self._loop_thread.start()

    @endpoint
    def get_weights_history(self):
        return list(self.weights_histories)

    def get_scores(self, sample_inputs: List[SampleInput]):
        scores = []
        for sample_input in sample_inputs:
            try:
                score = self.calculate_score(sample_input.image, sample_input.prompt)
                scores.append(normalize_score(score))
            except Exception as e:
                logger.error(e)
                scores.append(0.0)
        return scores

    def get_weights(self, uids: List[int], scores: List[float]):
        return weight_score(uids, scores)

    def get_metadata(self) -> dict:
        return {
            "model": self.model.get_metadata(),
            "dataset": self.dataset.get_metadata(),
        }

    def serve(self):
        """Start serving the validator on the specified host and port."""
        from communex.module.server import ModuleServer
        import uvicorn

        self.start_validation_loop()

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()
        uvicorn.run(app, host=self.settings.host, port=self.settings.port)


if __name__ == "__main__":
    settings = ValidatorSettings(use_testnet=True)
    Validator(key=classic_load_key("mosaic-validator0"), settings=settings).serve()
