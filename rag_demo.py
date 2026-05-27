import os
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

DATA_DIR = Path("sample_docs")


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n[page {page_no}]\n{text}")
    return "\n".join(pages)


def read_file(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def load_documents(folder=DATA_DIR):
    docs = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        text = read_file(path)
        if text.strip():
            docs.append({"source": path.name, "text": text})

    if not docs:
        raise RuntimeError(f"No documents found in {folder}. Add PDF/TXT/MD files first.")
    return docs


def chunk_text(text, chunk_size=1200, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def make_chunks(documents, chunk_size, overlap):
    chunks = []
    for doc in documents:
        for i, part in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": part})
    return chunks


class SentenceTransformerEmbedder:
    def __init__(self, model_name):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SentenceTransformer device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
            device=self.device,
        )
        return np.asarray(vectors, dtype="float32")


def make_embedder(cfg):
    return SentenceTransformerEmbedder(cfg["embedding_model"])


def build_faiss_index(embeddings):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("Using FAISS for vector search.")
    return index


def retrieve(question, index, chunks, embedder, top_k=4):
    q_emb = embedder.transform([question])
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[int(idx)].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def make_prompt(question, retrieved):
    context_blocks = []
    for i, chunk in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']}, chunk {chunk['chunk_id']}, score={chunk['score']:.3f}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)
    return f"""
You are a careful research assistant.
Answer the question using only the context below.
If the context does not contain the answer, say that the documents do not provide enough information.
Cite sources like [Source 1] when possible.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def fallback_answer(retrieved):
    lines = [
        "GOOGLE_API_KEY is not set, so I cannot call Gemini.",
        "Here are the most relevant retrieved excerpts instead:\n",
    ]
    for i, chunk in enumerate(retrieved[:2], start=1):
        excerpt = chunk["text"].replace("\n", " ")[:700]
        lines.append(f"[Source {i}: {chunk['source']}] {excerpt}...")
    return "\n\n".join(lines)


def gemini_answer(prompt, model_name, retrieved=None):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return fallback_answer(retrieved or [])

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text or "No answer returned by Gemini."
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return (
                "Gemini API quota was exceeded, so I am showing the retrieved excerpts instead.\n\n"
                + fallback_answer(retrieved or [])
            )
        return (
            "Gemini API call failed, so I am showing the retrieved excerpts instead.\n\n"
            + fallback_answer(retrieved or [])
        )


def build_rag_system(cfg):
    print("Loading documents...")
    docs = load_documents(DATA_DIR)

    print("Splitting documents into chunks...")
    chunks = make_chunks(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    print(f"Loaded {len(docs)} document(s), created {len(chunks)} chunk(s).")

    embedder = make_embedder(cfg)
    print("Creating Sentence Transformer embeddings...")
    embeddings = embedder.fit_transform([c["text"] for c in chunks])

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    return chunks, index, embedder


def main():
    load_dotenv()
    cfg = load_config()
    chunks, index, embedder = build_rag_system(cfg)

    print("\nReady. Ask a question, or type 'exit'.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(question, index, chunks, embedder, top_k=cfg["top_k"])

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(retrieved, start=1):
            print(f"{i}. {chunk['source']} | chunk {chunk['chunk_id']} | score={chunk['score']:.3f}")

        prompt = make_prompt(question, retrieved)
        answer = gemini_answer(prompt, cfg["llm_model"], retrieved=retrieved)

        print("\nAnswer:")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
