"""Gemini embeddings wrapper (Vertex).

Embeddings are L2-normalized so cosine similarity is a plain dot product. We
batch and retry; vectors are returned as float32 numpy arrays.
"""

from __future__ import annotations

import time
import random

import numpy as np

from .config import Config


class Embedder:
    def __init__(self, cfg: Config):
        from google import genai
        from google.genai.types import HttpOptions, EmbedContentConfig

        import os

        self._cfg = cfg
        self._EmbedContentConfig = EmbedContentConfig
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            # Gemini Developer API (aistudio.google.com): just an API key, no GCP project.
            self.client = genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
        else:
            # Vertex AI (the paper's runs): GOOGLE_CLOUD_PROJECT + application-default credentials.
            self.client = genai.Client(
                vertexai=True,
                project=cfg.vertex.project,
                location=cfg.vertex.location,
                http_options=HttpOptions(api_version="v1"),
            )

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        last_err = None
        for attempt in range(self._cfg.model.max_retries):
            try:
                resp = self.client.models.embed_content(
                    model=self._cfg.embed.name,
                    contents=texts,
                    config=self._EmbedContentConfig(
                        task_type=self._cfg.embed.task_type,
                        output_dimensionality=self._cfg.embed.dim,
                    ),
                )
                vecs = np.asarray([e.values for e in resp.embeddings], dtype=np.float32)
                return _l2_normalize(vecs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError(f"embed failed after retries: {last_err}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._cfg.embed.dim), dtype=np.float32)
        out = []
        bs = self._cfg.embed.batch_size
        for i in range(0, len(texts), bs):
            out.append(self._embed_batch(texts[i : i + bs]))
        return np.concatenate(out, axis=0)

    def embed_one(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)
