"""In-process text embedders. Nothing leaves the machine.

``model2vec`` (the default) runs a static embedding model: a lookup table plus
mean pooling, no PyTorch, no GPU, milliseconds per server. The weights are
downloaded once from the Hugging Face Hub at a pinned revision and cached.

``lexical`` is a TF-IDF fallback with no download at all, for air-gapped CI.
It only sees shared words, so it misses paraphrases.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from mcp_contrast.names import stem, tokenize

Matrix = npt.NDArray[np.float32]

DEFAULT_MODEL = "minishlab/potion-base-8M"
DEFAULT_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
_MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "modules.json"]


class EmbedderError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> Matrix:
        """Return one L2-normalized row per text."""
        ...


def _normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Model2VecEmbedder:
    name = "model2vec"

    def __init__(self, model: str = DEFAULT_MODEL, revision: str | None = DEFAULT_REVISION):
        self.model_id = model
        self.revision = revision
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        logging.getLogger("model2vec").setLevel(logging.ERROR)
        from model2vec import StaticModel

        path = Path(self.model_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            try:
                path = Path(
                    snapshot_download(
                        self.model_id, revision=self.revision, allow_patterns=_MODEL_FILES
                    )
                )
            except Exception as exc:
                raise EmbedderError(
                    f"could not load embedding model {self.model_id}@{self.revision}: {exc}\n"
                    "  Offline? Pre-download it, point --model at a local directory, "
                    "or use --embedder lexical."
                ) from exc
        self._model = StaticModel.from_pretrained(str(path))
        return self._model

    def embed(self, texts: list[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = np.asarray(self._load().encode(texts), dtype=np.float32)
        return _normalize(vectors)


class LexicalEmbedder:
    name = "lexical"

    def embed(self, texts: list[str]) -> Matrix:
        docs = [Counter(stem(t) for t in tokenize(text)) for text in texts]
        vocab = sorted({term for doc in docs for term in doc})
        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
        index = {term: i for i, term in enumerate(vocab)}
        df = Counter(term for doc in docs for term in doc)
        n = len(docs)
        matrix = np.zeros((n, len(vocab)), dtype=np.float32)
        for row, doc in enumerate(docs):
            for term, count in doc.items():
                idf = math.log((1 + n) / (1 + df[term])) + 1.0
                matrix[row, index[term]] = (1.0 + math.log(count)) * idf
        return _normalize(matrix)


def make_embedder(kind: str, model: str, revision: str | None) -> Embedder:
    if kind == "model2vec":
        return Model2VecEmbedder(model, revision)
    if kind == "lexical":
        return LexicalEmbedder()
    raise EmbedderError(f"unknown embedder {kind!r} (expected 'model2vec' or 'lexical')")
