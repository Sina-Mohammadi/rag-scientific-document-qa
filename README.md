# RAG Scientific Document QA

A compact Retrieval-Augmented Generation demo for asking questions over local PDFs.

This project is intentionally small and easy to explain. It includes three ready-made sample PDFs, so you can run the demo immediately after setup.

## What is included

- Three sample PDFs in `sample_docs/`
- PDF text extraction with `pypdf`
- Overlapping text chunks
- Sentence Transformer embeddings for semantic retrieval
- FAISS vector search
- Gemini API answer generation
- Simple retrieval/answer evaluation
- A few basic pytest tests

## Project structure

```text
rag-scientific-document-qa/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.yaml
├── rag_demo.py
├── evaluate_rag.py
├── create_sample_pdfs.py
├── sample_docs/
│   ├── fermentation_screening_report.pdf
│   ├── precision_agriculture_sensor_notes.pdf
│   └── reproducible_ml_project_notes.pdf
├── examples/
└── tests/
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Gemini API key

This project can use the Gemini API for answer generation. To use it, you need a Gemini API key.

Create or view your key here:

https://aistudio.google.com/apikey

Google’s Gemini documentation says that the Gemini API requires an API key and that API keys can be created and managed from the Google AI Studio API Keys page.

After creating the key, create a local `.env` file in the project folder:

```text
GOOGLE_API_KEY=your_api_key_here

## Run the demo

```bash
python rag_demo.py
```

Try these questions:

```text
Which strain was used with wheat bran?
At what temperature was Aspergillus oryzae grown?
What variables are useful as machine learning features in fermentation?
Which sensors are mentioned for crop monitoring?
How should experiments be made reproducible?
```

## Run evaluation

```bash
python evaluate_rag.py
```

It creates `rag_eval_results.csv` locally. This file is ignored by Git.

## Run tests

```bash
pytest
```

## GPU server notes

This version uses Sentence Transformer embeddings only. On a Linux GPU server, install a CUDA-enabled PyTorch version first, then run the project. If CUDA is available, the embedding model will use the GPU automatically. FAISS is used for vector search.

## Regenerate the sample PDFs

The PDFs are already included, but you can regenerate them with:

```bash
python create_sample_pdfs.py
```

## Notes

This is a learning/demo project, not a production system. A production system would need persistent indexes, stronger document parsing, more evaluation examples, privacy controls, monitoring, and deployment infrastructure.
