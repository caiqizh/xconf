"""Configuration dataclasses + YAML loading.

Defaults: Vertex project from ``GOOGLE_CLOUD_PROJECT``, ``us-central1``, ``gemini-2.5-flash`` for the model under
test and ``gemini-embedding-001`` (1536-dim) for retrieval keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import yaml


@dataclass
class VertexConfig:
    project: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))


@dataclass
class ModelConfig:
    name: str = "gemini-2.5-flash"  # model under test M
    temperature: float = 0.0
    max_retries: int = 8  # retryable (429/5xx/timeout) attempts with exp backoff
    max_workers: int = 8  # concurrent API calls during Stage A
    max_output_tokens: int = 1024
    # gemini-2.5 "thinking": default DISABLED (budget 0). Set to None to use the
    # model default, or a positive budget to allow thinking. (Defines a distinct model M.)
    thinking_budget: int | None = 0
    min_interval_s: float = 0.0  # optional global min spacing between requests (0=off)


@dataclass
class EmbedConfig:
    name: str = "gemini-embedding-001"
    dim: int = 1536
    task_type: str = "RETRIEVAL_DOCUMENT"
    batch_size: int = 16
    include_output: bool = True  # not used by the pipeline (question-only key)


@dataclass
class Config:
    vertex: VertexConfig = field(default_factory=VertexConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    seed: int = 0

    @classmethod
    def load(cls, path: str | None) -> "Config":
        cfg = cls()
        if not path:
            return cfg._apply_env()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section, sub in raw.items():
            if hasattr(cfg, section) and isinstance(sub, dict):
                cur = getattr(cfg, section)
                for k, v in sub.items():
                    if hasattr(cur, k):
                        setattr(cur, k, v)
            elif hasattr(cfg, section):
                setattr(cfg, section, sub)
        return cfg._apply_env()

    def _apply_env(self) -> "Config":
        # XCONF_LOCATION: spread lanes across Vertex quota pools (quota is per project x region; the
        # global endpoint is a separate pool -> parallel lanes on different endpoints ~double throughput)
        loc = os.environ.get("XCONF_LOCATION")
        if loc:
            self.vertex.location = loc
        return self

    def to_dict(self) -> dict:
        return asdict(self)
