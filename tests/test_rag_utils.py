from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rag_demo import chunk_text, load_documents, read_pdf


def test_chunk_text_creates_multiple_chunks():
    text = "a" * 3000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_sample_pdfs_are_readable():
    pdfs = list(Path("sample_docs").glob("*.pdf"))
    assert len(pdfs) >= 3
    text = "\n".join(read_pdf(p) for p in pdfs)
    assert "Lactobacillus" in text
    assert "plantarum" in text
    assert "Sentinel-2" in text


def test_load_documents_finds_pdfs():
    docs = load_documents(Path("sample_docs"))
    assert len(docs) >= 3
    assert any(d["source"].endswith(".pdf") for d in docs)
