"""Tests for SentenceTransformerEmbedder using a stub model (no real download)."""

from __future__ import annotations

import sys

import pytest

from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder


class StubModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer.

    Records every ``encode`` call so tests can assert on batching, the
    query instruction prefix, and dimension without touching the network.

    The base stub exposes only the deprecated ``get_sentence_embedding_dimension``
    alias, so the embedder's fallback path is exercised by every test that uses
    it directly.
    """

    def __init__(self, name: str = "stub", device: str = "cpu", **kwargs) -> None:
        self.name = name
        self.device = device
        self.init_kwargs = kwargs
        self.calls: list[tuple[list[str], dict]] = []
        self._dimension = 8

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [[float(i)] * self._dimension for i in range(len(texts))]


class StubModelNewApi(StubModel):
    """Stub exposing only the current ``get_embedding_dimension`` API.

    Newer sentence-transformers releases renamed the method and emit a
    FutureWarning when the old alias is used, so the embedder must prefer the
    new name when it is present. The deprecated alias is deliberately absent so
    a test can prove the embedder never calls it.
    """

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_sentence_embedding_dimension(self) -> int:
        raise AttributeError("StubModelNewApi deliberately hides the deprecated alias")


class StubModelOldApi(StubModel):
    """Stub exposing only the deprecated ``get_sentence_embedding_dimension``.

    Older sentence-transformers releases predate ``get_embedding_dimension``;
    the embedder must fall back to the old alias rather than crash.
    """

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def get_embedding_dimension(self) -> int:
        raise AttributeError("StubModelOldApi deliberately hides the new API")


@pytest.fixture
def stub_module(monkeypatch):
    """Inject a fake ``sentence_transformers`` module into sys.modules."""
    return _inject_stub(monkeypatch, StubModel)


@pytest.fixture
def stub_module_new_api(monkeypatch):
    """Inject a stub that exposes only the current ``get_embedding_dimension``."""
    return _inject_stub(monkeypatch, StubModelNewApi)


@pytest.fixture
def stub_module_old_api(monkeypatch):
    """Inject a stub that exposes only the deprecated alias."""
    return _inject_stub(monkeypatch, StubModelOldApi)


def _inject_stub(monkeypatch, model_cls):
    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = model_cls
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


class TestSentenceTransformerEmbedder:
    def test_constructs_with_stub(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.model_name == "BAAI/bge-small-en-v1.5"
        assert e.dimension == 8
        assert e._model.init_kwargs["local_files_only"] is True

    def test_missing_model_fails_without_download_fallback(self, monkeypatch):
        class MissingLocalModel:
            def __init__(self, name, device, *, local_files_only):
                assert local_files_only is True
                raise OSError("model files are not cached")

        _inject_stub(monkeypatch, MissingLocalModel)
        with pytest.raises(OSError, match="not cached"):
            SentenceTransformerEmbedder()

    def test_dimension_from_model(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.dimension == e._model.get_sentence_embedding_dimension()

    def test_query_instruction_logic(self, stub_module):
        e = SentenceTransformerEmbedder(instruction="SEARCH: ")
        e.embed_query("hello")
        e.embed_documents(["hello"])

        query_call = e._model.calls[0][0]
        doc_call = e._model.calls[1][0]
        assert query_call == ["SEARCH: hello"]
        assert doc_call == ["hello"]

    def test_query_instruction_default(self, stub_module):
        e = SentenceTransformerEmbedder()
        e.embed_query("hello")
        assert e._model.calls[0][0] == [
            "Represent this sentence for searching relevant passages: hello"
        ]

    def test_no_instruction(self, stub_module):
        e = SentenceTransformerEmbedder(instruction="")
        e.embed_query("hello")
        assert e._model.calls[0][0] == ["hello"]

    def test_batching(self, stub_module):
        e = SentenceTransformerEmbedder(batch_size=2)
        e.embed_documents(["a", "b", "c", "d", "e"])
        assert len(e._model.calls) == 3  # 2 + 2 + 1
        assert e._model.calls[0][0] == ["a", "b"]
        assert e._model.calls[2][0] == ["e"]

    def test_embed_documents_empty(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.embed_documents([]) == []

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_batch_size_must_be_positive(self, stub_module, batch_size):
        with pytest.raises(ValueError, match="positive integer"):
            SentenceTransformerEmbedder(batch_size=batch_size)

    @pytest.mark.parametrize("batch_size", [True, 1.5, "2"])
    def test_batch_size_must_be_integer(self, stub_module, batch_size):
        with pytest.raises(TypeError, match="positive integer"):
            SentenceTransformerEmbedder(batch_size=batch_size)

    def test_empty_query_is_rejected(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(ValueError, match="non-empty string"):
            embedder.embed_query("  ")

    def test_empty_document_text_is_rejected(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(ValueError, match="non-empty strings"):
            embedder.embed_documents(["valid", " "])

    def test_document_texts_must_be_a_list(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(TypeError, match="provided as a list"):
            embedder.embed_documents("malformed")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("output", "error"),
        [
            ([[1.0] * 8], "unexpected number"),
            ([[1.0] * 7, [1.0] * 7], "wrong dimension"),
            ([[float("nan")] * 8, [0.0] * 8], "non-finite"),
            ([[float("inf")] * 8, [0.0] * 8], "non-finite"),
            ([["not-a-number"] * 8, [0.0] * 8], "non-numeric"),
            ([[True] * 8, [0.0] * 8], "non-numeric"),
            ([[False] * 8, [0.0] * 8], "non-numeric"),
            ([["1"] * 8, [0.0] * 8], "non-numeric"),
            ([[(1 + 0j)] * 8, [0.0] * 8], "non-numeric"),
            ([[float("-inf")] * 8, [0.0] * 8], "non-finite"),
        ],
    )
    def test_invalid_model_output_fails_closed(self, stub_module, output, error):
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: output
        with pytest.raises(RuntimeError, match=error):
            embedder.embed_documents(["synthetic input", "second input"])

    def test_query_output_is_dimension_and_finiteness_checked(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [[float("nan")] * 8]
        with pytest.raises(RuntimeError, match="non-finite"):
            embedder.embed_query("synthetic query")

    def test_python_integer_and_float_coordinates_are_supported(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [
            list(range(8)),
            [float(i) + 0.5 for i in range(8)],
        ]
        vectors = embedder.embed_documents(["integer vector", "float vector"])
        assert vectors[0] == [float(i) for i in range(8)]
        assert vectors[1] == [float(i) + 0.5 for i in range(8)]

    def test_torch_scalar_output_is_supported_without_importing_torch(self, stub_module):
        class Tensor:
            __module__ = "torch"

            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [[Tensor(float(i)) for i in range(8)]]
        assert embedder.embed_documents(["synthetic tensor vector"])[0] == [
            float(i) for i in range(8)
        ]

    def test_torch_boolean_scalar_is_rejected(self, stub_module):
        class Tensor:
            __module__ = "torch"

            def item(self):
                return True

        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [[Tensor() for _ in range(8)]]
        with pytest.raises(RuntimeError, match="non-numeric"):
            embedder.embed_documents(["synthetic malformed vector"])

    @pytest.mark.parametrize(
        ("dimension", "error_type"),
        [
            (True, TypeError),
            (False, TypeError),
            (3.9, ValueError),
            (0, ValueError),
            (-1, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
            ("8", TypeError),
        ],
    )
    def test_invalid_model_dimensions_are_rejected(self, stub_module, dimension, error_type):
        class InvalidDimensionModel(StubModel):
            def get_embedding_dimension(self):
                return dimension

        stub_module.SentenceTransformer = InvalidDimensionModel
        with pytest.raises(error_type, match="positive integral vector dimension"):
            SentenceTransformerEmbedder()

    @pytest.mark.parametrize("dimension", [8, 8.0])
    def test_valid_python_dimensions_are_preserved(self, stub_module, dimension):
        class ValidDimensionModel(StubModel):
            def get_embedding_dimension(self):
                return dimension

        stub_module.SentenceTransformer = ValidDimensionModel
        assert SentenceTransformerEmbedder().dimension == 8

    def test_valid_numpy_dimensions_and_coordinates_are_supported(self, stub_module):
        np = pytest.importorskip("numpy")

        class NumpyDimensionModel(StubModel):
            def get_embedding_dimension(self):
                return np.int64(8)

        stub_module.SentenceTransformer = NumpyDimensionModel
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [
            [np.int64(i) for i in range(8)],
            [np.float32(i / 2) for i in range(8)],
        ]
        vectors = embedder.embed_documents(["integer vector", "float vector"])
        assert vectors[0] == [float(i) for i in range(8)]
        assert vectors[1] == [float(i / 2) for i in range(8)]

    def test_numpy_integral_float_dimension_is_supported(self, stub_module):
        np = pytest.importorskip("numpy")

        class NumpyFloatDimensionModel(StubModel):
            def get_embedding_dimension(self):
                return np.float64(8.0)

        stub_module.SentenceTransformer = NumpyFloatDimensionModel
        assert SentenceTransformerEmbedder().dimension == 8

    def test_numpy_boolean_dimension_and_vector_values_are_rejected(self, stub_module):
        np = pytest.importorskip("numpy")

        class NumpyBooleanDimensionModel(StubModel):
            def get_embedding_dimension(self):
                return np.bool_(True)

        stub_module.SentenceTransformer = NumpyBooleanDimensionModel
        with pytest.raises(TypeError, match="positive integral vector dimension"):
            SentenceTransformerEmbedder()

        stub_module.SentenceTransformer = StubModel
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [[np.bool_(True)] * 8]
        with pytest.raises(RuntimeError, match="non-numeric"):
            embedder.embed_query("synthetic query")

    def test_boolean_query_vector_is_rejected(self, stub_module):
        embedder = SentenceTransformerEmbedder()
        embedder._model.encode = lambda *args, **kwargs: [[True] * 8]
        with pytest.raises(RuntimeError, match="non-numeric"):
            embedder.embed_query("synthetic query")

    def test_model_name_override(self, stub_module):
        e = SentenceTransformerEmbedder(model_name="other-model")
        assert e.model_name == "other-model"
        assert e._model.name == "other-model"

    def test_missing_extra_raises_clear_error(self, monkeypatch):
        """Without sentence-transformers the constructor must fail with guidance."""
        blocker = ModuleBlocker("sentence_transformers")
        monkeypatch.setitem(sys.modules, "sentence_transformers", blocker)
        with pytest.raises(RuntimeError, match="sentence-transformers is required"):
            SentenceTransformerEmbedder()

    def test_normalise_embeddings_passed_through(self, stub_module):
        e = SentenceTransformerEmbedder(normalize_embeddings=True)
        e.embed_query("x")
        assert e._model.calls[0][1]["normalize_embeddings"] is True

    def test_dimension_prefers_new_api(self, stub_module_new_api):
        """The embedder must use ``get_embedding_dimension`` when present."""
        e = SentenceTransformerEmbedder()
        assert e.dimension == 8
        assert e._model.get_embedding_dimension() == 8

    def test_dimension_falls_back_to_old_api(self, stub_module_old_api):
        """The embedder must fall back to the deprecated alias."""
        e = SentenceTransformerEmbedder()
        assert e.dimension == 8
        assert e._model.get_sentence_embedding_dimension() == 8

    def test_dimension_does_not_call_deprecated_alias_when_new_api_present(
        self, stub_module_new_api
    ):
        """A stub that hides the old alias must still construct successfully."""
        e = SentenceTransformerEmbedder()
        assert e.dimension == 8


class ModuleBlocker:
    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, item: str):
        raise ImportError(
            "sentence-transformers is required for the real embedder; "
            'install with: python -m pip install -e ".[embeddings]"'
        )
