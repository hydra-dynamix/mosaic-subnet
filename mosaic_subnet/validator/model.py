from io import BytesIO

import hpsv2
import torch
from PIL import Image
from communex.module.module import Module
from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline


class CLIP(Module):
    def __init__(self, model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K") -> None:
        super().__init__()
        self.model_name = model_name
        logger.info(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_similarity(self, file: bytes, prompt: str) -> float:
        image = Image.open(BytesIO(file))
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt", padding=True
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits_per_image.sum().tolist()

    def get_metadata(self) -> dict:
        return {
            "model": self.model_name,
            "device": str(self.device),
            "requirements": {
                "min_ram": "8GB",  # Minimum RAM needed
                "min_vram": "4GB",  # Minimum VRAM if using GPU
                "gpu_optional": True,  # Can run on CPU
                "inference_type": "scoring",  # Just scoring, not generation
                "avg_inference_time": "~100ms"  # Fast inference
            }
        }


class HPS(Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "HPSv2"
        logger.info(self.model_name)

    def get_similarity(self, file: bytes, prompt: str) -> float:
        img = Image.open(BytesIO(file)).convert("RGB")
        results = hpsv2.score(img, prompt=prompt, hps_version="v2.1")
        return results[0] * 100

    def get_metadata(self) -> dict:
        return {
            "model": self.model_name,
            "requirements": {
                "min_ram": "4GB",  # Lower RAM requirements
                "min_vram": "2GB",  # Lower VRAM if using GPU
                "gpu_optional": True,  # Can run on CPU
                "inference_type": "scoring",  # Just scoring, not generation
                "avg_inference_time": "~50ms"  # Very fast inference
            }
        }


class NSFWChecker(Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "Falconsai/nsfw_image_detection"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=self.device,
        )

    def check_nsfw(self, file: bytes) -> bool:
        image = Image.open(BytesIO(file))
        for c in self.classifier(image):
            if c["label"] == "nsfw" and c["score"] > 0.8:
                return True
        return False

    def get_metadata(self) -> dict:
        return {
            "model": self.model_name,
            "device": str(self.device),
            "requirements": {
                "min_ram": "8GB",  # Minimum RAM needed
                "min_vram": "4GB",  # Minimum VRAM if using GPU
                "gpu_optional": True,  # Can run on CPU
                "inference_type": "classification",  # Classification task
                "avg_inference_time": "~150ms"  # Moderate inference time
            }
        }


if __name__ == "__main__":
    import httpx

    resp = httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = resp.content
    c = CLIP()
    score_cat = c.get_similarity(file=image, prompt="cat")
    score_dog = c.get_similarity(file=image, prompt="dog")
    print(score_cat, score_dog)

    nc = NSFWChecker()
    is_nsfw = nc.check_nsfw(image)
    print(is_nsfw)
