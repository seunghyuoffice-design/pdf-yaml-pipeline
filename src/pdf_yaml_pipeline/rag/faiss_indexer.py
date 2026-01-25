# SPDX-License-Identifier: MIT
import json
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


class FaissIndexer:
    def __init__(self, model: str, use_gpu: bool = True):
        self.model = SentenceTransformer(model)
        self.use_gpu = use_gpu

    def build(self, jsonl: Path, out: Path, role: str = "canonical", batch_size: int = 256):
        metas: list[dict] = []
        index = None
        batch_texts = []

        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                if j["meta"].get("role") != role:
                    continue
                batch_texts.append(j["text"])
                metas.append(j["meta"])

                if len(batch_texts) >= batch_size:
                    vecs = self.model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
                    if index is None:
                        index = faiss.IndexFlatIP(vecs.shape[1])
                        if self.use_gpu:
                            try:
                                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                            except Exception:
                                pass  # fallback to CPU
                    index.add(vecs.astype("float32"))
                    batch_texts = []

        if batch_texts:
            vecs = self.model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            if index is None:
                index = faiss.IndexFlatIP(vecs.shape[1])
                if self.use_gpu:
                    try:
                        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                    except Exception:
                        pass  # fallback to CPU
            index.add(vecs.astype("float32"))

        if index is None:
            raise ValueError(f"No chunks with role={role}")
        out.mkdir(parents=True, exist_ok=True)

        # Save CPU index
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index

        faiss.write_index(cpu_index, str(out / "index.faiss"))
        pickle.dump(metas, (out / "index_metadata.pkl").open("wb"))
        with (out / "index_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, default=_json_default)

        return {"indexed": len(metas), "dim": index.d}


def _json_default(value):
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    return str(value)
