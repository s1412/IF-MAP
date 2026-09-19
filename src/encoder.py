"""Cached prompt embeddings; Qwen3 defaults to 512 dimensions."""

import hashlib
import os
import time
from typing import List

import numpy as np

from src.errors import fail, report_invalid

QWEN3_MODEL = "qwen/qwen3-embedding-8b"
QWEN3_DIMENSIONS = 512
DEFAULT_LOCAL_SERVER = "http://localhost:8400/v1/embeddings"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    zero = (norms == 0).flatten()
    if zero.any():
        report_invalid("encoder._l2_normalize",
                       "zero-norm embedding(s) encountered before normalize",
                       num_zero=int(zero.sum()))
        norms[norms == 0] = 1.0
    return vecs / norms


class Encoder:
    """Cached embedding backend."""

    def __init__(self, encoder_name: str):
        self.name = encoder_name
        self._cache: dict = {}


        if encoder_name in ("qwen/qwen3-embedding-8b", "qwen3", "qwen"):
            self._backend = os.environ.get("IFIP_EMBEDDING_MODE", "openrouter")
            self._model = QWEN3_MODEL
            self._dim = QWEN3_DIMENSIONS
            self._sbert = None
            if self._backend not in ("openrouter", "local_server"):
                fail("encoder.__init__",
                     "Qwen3 encoder requires IFIP_EMBEDDING_MODE in "
                     "{openrouter, local_server}",
                     IFIP_EMBEDDING_MODE=self._backend)
        elif encoder_name == "hash_dev":
            self._backend = "hash_dev"
            self._model = None
            self._dim = QWEN3_DIMENSIONS
            self._sbert = None
        elif encoder_name.startswith("text-embedding-"):
            self._backend = "openai"
            self._model = encoder_name
            self._dim = 3072 if "large" in encoder_name else 1536
            self._sbert = None
        else:

            self._backend = "sbert"
            self._model = encoder_name
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(encoder_name)
            self._dim = self._sbert.get_sentence_embedding_dimension()

        print(f"[Encoder] backend={self._backend}  model={self._model}  dim={self._dim}")


    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        if self._backend == "openrouter":
            vecs = _encode_openrouter(texts, self._model, self._dim)
        elif self._backend == "local_server":
            vecs = _encode_local_server(texts, self._dim)
        elif self._backend == "openai":
            vecs = _encode_openai(texts, self._model)
        elif self._backend == "sbert":
            vecs = self._sbert.encode(texts, convert_to_numpy=True,
                                      show_progress_bar=False).astype(np.float32)
        elif self._backend == "hash_dev":
            vecs = _encode_hash_dev(texts, self._dim)
        else:
            fail("encoder._encode_raw", "unknown backend", backend=self._backend)

        if vecs.shape[0] != len(texts):
            fail("encoder._encode_raw",
                 "embedding count mismatch (some inputs lost)",
                 expected=len(texts), got=int(vecs.shape[0]), backend=self._backend)
        return _l2_normalize(vecs)

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._cache:
            return self._cache[h]
        vec = self._encode_raw([text])[0]
        self._cache[h] = vec
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        out = [None] * len(texts)
        todo_idx, todo_txt = [], []
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if h in self._cache:
                out[i] = self._cache[h]
            else:
                todo_idx.append(i)
                todo_txt.append(t)
        if todo_txt:
            vecs = self._encode_raw(todo_txt)
            for j, i in enumerate(todo_idx):
                h = hashlib.sha256(todo_txt[j].encode("utf-8")).hexdigest()
                self._cache[h] = vecs[j]
                out[i] = vecs[j]
        return np.stack(out, axis=0)

    @property
    def dim(self) -> int:
        return self._dim


def _encode_openrouter(texts: List[str], model: str, dim: int,
                       batch_size: int = 20, max_retries: int = 6) -> np.ndarray:
    """Request OpenRouter embeddings."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        fail("encoder._encode_openrouter",
             "OPENROUTER_API_KEY not set (required for Qwen3 openrouter mode)")

    session = requests.Session()

    proxy = (os.environ.get("IFIP_HTTP_PROXY")
             or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })

    all_emb = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for bi in range(0, len(texts), batch_size):
        batch = texts[bi:bi + batch_size]
        body = {"model": model, "input": batch, "dimensions": dim}
        for attempt in range(max_retries):
            try:
                resp = session.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    json=body, timeout=120,
                )
                if resp.status_code != 200:
                    if resp.status_code >= 500 and attempt < max_retries - 1:
                        time.sleep(15 * (2 ** attempt))
                        continue
                    fail("encoder._encode_openrouter",
                         f"OpenRouter API HTTP {resp.status_code}",
                         body=resp.text[:200], batch=f"{bi//batch_size+1}/{total_batches}")
                items = sorted(resp.json()["data"], key=lambda x: x["index"])
                all_emb.extend(it["embedding"] for it in items)
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(15 * (2 ** attempt))
                else:
                    fail("encoder._encode_openrouter",
                         f"network failure after {max_retries} retries: "
                         f"{type(e).__name__}: {e}",
                         batch=f"{bi//batch_size+1}/{total_batches}")
        if bi // batch_size < total_batches - 1:
            time.sleep(1.0)

    return np.array(all_emb, dtype=np.float32)


def _encode_local_server(texts: List[str], dim: int,
                         batch_size: int = 32) -> np.ndarray:
    """Request local embeddings."""
    import requests

    url = os.environ.get("IFIP_EMBEDDING_SERVER", DEFAULT_LOCAL_SERVER)
    all_emb = []
    for bi in range(0, len(texts), batch_size):
        batch = texts[bi:bi + batch_size]
        try:
            resp = requests.post(url, json={"model": "default", "input": batch},
                                 timeout=120)
        except requests.exceptions.ConnectionError as e:
            fail("encoder._encode_local_server",
                 f"cannot connect to local embedding server (is it running?): {e}",
                 url=url)
        if resp.status_code != 200:
            fail("encoder._encode_local_server",
                 f"local server HTTP {resp.status_code}", body=resp.text[:200], url=url)
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        all_emb.extend(it["embedding"] for it in items)

    vecs = np.array(all_emb, dtype=np.float32)
    if vecs.shape[1] < dim:
        fail("encoder._encode_local_server",
             "server embedding dim smaller than requested",
             server_dim=int(vecs.shape[1]), requested=dim)
    return vecs[:, :dim]


def _encode_openai(texts: List[str], model: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    if not os.environ.get("OPENAI_API_KEY"):
        fail("encoder._encode_openai", "OPENAI_API_KEY not set")
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _encode_hash_dev(texts: List[str], dim: int) -> np.ndarray:
    """Deterministic offline test embeddings."""
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
        out[i] = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return out


def init_encoder(encoder_name: str) -> Encoder:
    return Encoder(encoder_name)


def encode_prompt(encoder: Encoder, text: str) -> np.ndarray:
    return encoder.encode(text)


def encode_batch(encoder: Encoder, texts: List[str]) -> np.ndarray:
    return encoder.encode_batch(texts)


def get_embedding_dim(encoder: Encoder) -> int:
    return encoder.dim
