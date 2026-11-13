"""Real embedder backed by sentence-transformers (BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

import math
import numbers
import os

from rag_agent.embeddings.protocol import Embedder

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# bge-small-en-v1.5 accepts up to 512 WordPiece tokens including special tokens.
_MAX_MODEL_TOKENS = 512


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a Hugging Face sentence-transformers model.

    The ``sentence-transformers`` package is imported lazily so that the base
    package installs and imports without it. Constructing this embedder without
    the ``[embeddings]`` extra installed raises a clear error.

    Model files are resolved from the local Hugging Face cache only. Missing
    artifacts fail instead of triggering a network download.

    Args:
        model_name: HF model id, overridable via the ``EMBEDDING_MODEL``
            environment variable.
        instruction: Prefix prepended to query texts. bge models are
            instruction-aware: documents are embedded raw while queries get
            this prefix so the model's query-task is consistent.
        batch_size: Number of texts embedded per model call.
        device: Torch device string; defaults to CPU.
        normalize_embeddings: L2-normalise every vector so cosine similarity
            equals the dot product in Qdrant.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = 32,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                'Install it with: python -m pip install -e ".[embeddings]"'
            ) from exc

        self._model_name = model_name or os.environ.get("EMBEDDING_MODEL", _DEFAULT_MODEL)
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        # Keep model access deterministic and offline by default. Missing model
        # artifacts must fail rather than trigger an implicit Hub download.
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            local_files_only=True,
        )
        # Prefer the current API; fall back to the deprecated alias so the
        # embedder keeps working with older sentence-transformers releases that
        # predate ``get_embedding_dimension``. The lookup is attribute-based but
        # the call is guarded so a stub that exposes only the alias is detected
        # by the AttributeError rather than by a callable check.
        try:
            raw_dimension = self._model.get_embedding_dimension()
        except AttributeError:
            raw_dimension = self._model.get_sentence_embedding_dimension()
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, numbers.Real):
            raise TypeError("Embedding model must report a positive integral vector dimension")
        if isinstance(raw_dimension, numbers.Integral):
            dimension = int(raw_dimension)
        elif (
            not math.isfinite(raw_dimension)
            or raw_dimension <= 0
            or raw_dimension != math.floor(raw_dimension)
        ):
            raise ValueError("Embedding model must report a positive integral vector dimension")
        else:
            dimension = int(raw_dimension)
        if dimension <= 0:
            raise ValueError("Embedding model must report a positive integral vector dimension")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("Document texts must be provided as a list")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Document texts must be non-empty strings")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings = self._model.encode(
                batch,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=False,
                show_progress_bar=False,
            )
            vectors.extend(self._validated_vectors(embeddings, len(batch)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a non-empty string")
        prefixed = self.instruction + text if self.instruction else text
        embeddings = self._model.encode(
            [prefixed],
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return self._validated_vectors(embeddings, 1)[0]

    def _validated_vectors(self, embeddings, expected_count: int) -> list[list[float]]:
        """Validate model output before vectors reach storage or retrieval."""
        try:
            rows = list(embeddings)
        except TypeError as exc:
            raise RuntimeError("Embedding model returned malformed vectors") from exc
        if len(rows) != expected_count:
            raise RuntimeError("Embedding model returned an unexpected number of vectors")

        vectors: list[list[float]] = []
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise RuntimeError("Embedding model returned malformed vectors")
            try:
                values = list(row)
            except TypeError as exc:
                raise RuntimeError("Embedding model returned malformed vectors") from exc
            validated_values = []
            for value in values:
                if isinstance(value, bool):
                    raise RuntimeError("Embedding model returned non-numeric vectors")
                if not isinstance(value, numbers.Real):
                    value_type = type(value)
                    if value_type.__module__ == "torch" and value_type.__name__ == "Tensor":
                        try:
                            value = value.item()
                        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "Embedding model returned malformed vectors"
                            ) from exc
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise RuntimeError("Embedding model returned non-numeric vectors")
                validated_values.append(value)
            try:
                vector = [float(value) for value in validated_values]
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Embedding model returned non-numeric vectors") from exc
            if len(vector) != self.dimension:
                raise RuntimeError("Embedding model returned a vector with the wrong dimension")
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("Embedding model returned non-finite vector values")
            vectors.append(vector)
        return vectors


def model_max_tokens() -> int:
    """Maximum WordPiece token length accepted by the bge embedding model."""
    return _MAX_MODEL_TOKENS
