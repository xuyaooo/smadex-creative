from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


class CLIPCreativeEncoder:
    """Visual encoder for creative assets.

    Despite the name, the underlying model is now SigLIP 2 (google/siglip2-base-patch16-256)
    by default. The class name and method surface are kept stable for backwards compatibility
    with the rest of the pipeline; the model is loaded via AutoModel / AutoProcessor so any
    HF vision-text dual-encoder (CLIP, SigLIP, SigLIP 2) can be plugged in via config.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-256", device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._torch = torch

    @staticmethod
    def _as_tensor(feats):
        import torch
        # transformers 5.x may wrap output in a dataclass
        if isinstance(feats, torch.Tensor):
            return feats
        if hasattr(feats, "image_embeds"):
            return feats.image_embeds
        if hasattr(feats, "text_embeds"):
            return feats.text_embeds
        if hasattr(feats, "pooler_output"):
            return feats.pooler_output
        if hasattr(feats, "last_hidden_state"):
            return feats.last_hidden_state[:, 0]
        return feats[0]

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        import torch
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            feats = self._as_tensor(self.model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze().cpu().numpy().astype(np.float32)

    def encode_batch(self, image_paths: List[str | Path], batch_size: int = 32) -> np.ndarray:
        import torch
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i: i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                feats = self._as_tensor(self.model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_feats, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            feats = self._as_tensor(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze().cpu().numpy().astype(np.float32)

    def compute_similarity(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
        return float(np.dot(img_emb, txt_emb))


class EmbeddingCache:
    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._embeddings: np.ndarray | None = None
        self._ids: List[int] | None = None

    def exists(self) -> bool:
        return self.cache_path.exists()

    def save(self, embeddings: np.ndarray, creative_ids: List[int]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, embeddings=embeddings, creative_ids=creative_ids)
        self._embeddings = embeddings
        self._ids = list(creative_ids)

    def load(self) -> Tuple[np.ndarray, List[int]]:
        data = np.load(self.cache_path)
        self._embeddings = data["embeddings"]
        self._ids = data["creative_ids"].tolist()
        return self._embeddings, self._ids

    def get_embedding(self, creative_id: int) -> np.ndarray | None:
        if self._embeddings is None or self._ids is None:
            self.load()
        if creative_id in self._ids:
            idx = self._ids.index(creative_id)
            return self._embeddings[idx]
        return None

    def get_all(self) -> Tuple[np.ndarray, List[int]]:
        if self._embeddings is None:
            self.load()
        return self._embeddings, self._ids
