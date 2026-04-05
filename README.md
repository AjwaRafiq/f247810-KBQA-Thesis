# BertT5-LLaMA-KBQA: WebQSP System README
**Student:** Ajwa Rafiq (24F-7810) | **Supervisor:** Dr. Muhammad Fayyaz | **NUCES CFD Campus**

---

## System Overview
A Knowledge Base Question Answering (KBQA) system that combines BERT, T5, and LLaMA-3-8B with Wikidata and DBpedia knowledge bases to answer natural language questions.

**Result:** F1 = 87.02% on WebQSP 
---

## Step 1: Data Preparation

```bash
# Preprocess WebQSP dataset
python scripts/preprocess_webqsp.py

# Pre-compute KB caches on master node (has internet)
python scripts/precompute_wikidata.py   # 979 entries, 641 with answers
python scripts/precompute_dbpedia.py    # 1517 entries, 476 with answers
```

**Output:**
- `data/webqsp/processed/train.json` (3,098 questions)
- `data/webqsp/processed/test.json`  (1,639 questions)
- `data/webqsp/wikidata_cache.json`
- `data/webqsp/dbpedia_cache.json`

---

## Step 2: Train BERT Ranker

```bash
sbatch scripts/train_bert_webqsp_job.sh
```

- **Model:** bert-base-uncased
- **Task:** Binary classification (correct/wrong S-expression)
- **Training samples:** 15,383
- **Result:** Accuracy = 96.19%
- **Saved to:** `models/bert_ranker/trained/`

---

## Step 3: Train T5 Generator

```bash
sbatch scripts/train_t5_webqsp_job.sh
```

- **Model:** t5-large
- **Task:** Generate S-expression from question + candidates
- **Training samples:** 3,044
- **Result:** Exact Match = 99.94%
- **Saved to:** `models/t5_generator/trained/`

---

## Step 4: Fine-tune LLaMA-3-8B

```bash
sbatch scripts/train_llama_webqsp_job.sh
```

- **Model:** Meta-Llama-3-8B-Instruct
- **Method:** LoRA (only 0.52% parameters trained)
- **Task:** Rerank KB answers
- **Training samples:** 3,021
- **Saved to:** `models/llama3/trained_webqsp/`

---

## Step 5: Evaluate

```bash
sbatch scripts/evaluate_webqsp_v6_job.sh
```

**Pipeline:**
```
Question
   ↓
BERT ranks candidates → T5 generates S-expression
   ↓
Ensemble selects best candidate
   ↓
KB Tier 1: WebQSP pre-executed answers
KB Tier 2: Wikidata SPARQL cache
KB Tier 3: DBpedia SPARQL cache
   ↓
LLaMA-3-8B reranks KB answers
   ↓
Final Answer
```

**Results saved to:** `results/webqsp/results_v6.json`

---

## Final Results

| Metric | Our System | 
|--------|-----------|
| F1 Score | **87.02%** |
| Relaxed EM | **99.00%** |
| Single-hop F1 | **89.09%** |
| Multi-hop F1 | **84.99%** | 

---

## Key Files

```
scripts/
├── preprocess_webqsp.py          # Data preprocessing
├── precompute_wikidata.py        # Wikidata KB cache
├── precompute_dbpedia.py         # DBpedia KB cache
├── train_bert_webqsp.py          # BERT training
├── train_t5_webqsp.py            # T5 training
├── train_llama_webqsp.py         # LLaMA LoRA training
├── evaluate_webqsp_v6.py         # Final evaluation
└── app_master.py                 # Demo Flask app

data/webqsp/
├── processed/train.json          # Training data
├── processed/test.json           # Test data
├── wikidata_cache.json           # Wikidata answers
└── dbpedia_cache.json            # DBpedia answers

models/
├── bert_ranker/trained/          # BERT model
├── t5_generator/trained/         # T5 model
└── llama3/trained_webqsp/        # LLaMA LoRA adapter

results/webqsp/
└── results_v6.json               # Final results
```

---

## Hardware Used

| Component | Hardware |
|-----------|----------|
| BERT Training | CPU nodes (compute3-5) |
| T5 Training | Tesla P40 GPU (24GB) |
| LLaMA Training | Tesla P40 GPU (24GB) |
| KB Cache | Master node (internet access) |
| Evaluation | Tesla P40 GPU (24GB) |

