# BERT-LLAMA-KBQA
This repository contains the implementation code of the proposed BERT-LLAMA-KBQA architecture. It includes all the major components of the framework.

## Environment Setup

| Library          | Version      |
|-------------------|--------------|
| torch             | 2.7.1+cu118  |
| transformers      | 4.47.0       |
| peft              | 0.13.0       |
| bitsandbytes      | 0.44.1       |
| accelerate        | 1.1.1        |
| datasets          | 3.1.0        |
| scikit-learn      | 1.5.2        |
| imbalanced-learn  | 0.12.3       |
| sentencepiece     | 0.2.0        |
| tokenizers        | 0.20.3       |
| requests          | 2.32.3       |
| numpy             | 1.26.4       |
| scipy             | 1.13.1       |
| matplotlib        | 3.10.8       |
| tqdm              | 4.66.6       |
| huggingface-hub   | 0.26.2       |
| protobuf          | 5.28.3       |

## KBQA Dataset

This work uses the **WebQuestionsSP (WebQSP)** dataset, a widely used benchmark for Knowledge Base Question Answering (KBQA).

The official WebQSP dataset can be downloaded from the Microsoft Download Center:
[WebQSP Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763)

## Freebase Setup

Dataset uses Freebase as the underlying knowledge base. Before running the project, you need to set up a Virtuoso triplestore to host the Freebase RDF data.

**Download Database:**

```bash
cd Freebase-Setup
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
tar -zxvf virtuoso_db.zip
```

**To start the Virtuoso service:**

```bash
python3 virtuoso.py start 3001 -d virtuoso_db
```

**To stop a currently running service at the same port:**

```bash
python3 virtuoso.py stop 3001
```


