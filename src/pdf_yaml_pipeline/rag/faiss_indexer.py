# SPDX-License-Identifier: MIT
from pathlib import Path
import json
import pickle

import faiss
from sentence_transformers import SentenceTransformer


class FaissIndexer:
    def __init__(self, model: str, use_gpu: bool = True):
        self.model = SentenceTransformer(model)
        self.use_gpu = use_gpu

    def build(self, jsonl: Path, out: Path, role: str = "canonical"):
        texts, metas = [], []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["meta"].get("role") == role:
                texts.append(j["text"])
                metas.append(j["meta"])

        if not texts:
            raise ValueError(f"No chunks with role={role}")

        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(vecs.shape[1])

        if self.use_gpu:
            try:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            except Exception:
                pass  # fallback to CPU

        index.add(vecs.astype("float32"))
        out.mkdir(parents=True, exist_ok=True)

        # Save CPU index
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index

        faiss.write_index(cpu_index, str(out / "index.faiss"))
        pickle.dump(metas, (out / "index_metadata.pkl").open("wb"))

        return {"indexed": len(texts), "dim": vecs.shape[1]}
