import json
import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ── config ─────────────────────────────────────────────────────────────────
BERT_MODEL_PATH = "/home/f247810/KBQA_WebQSP_improve/models/BERT_retriever/best"
DATA_DIR        = "/home/f247810/KBQA_WebQSP_improve/data/bert_input"
PROCESSED_DIR   = "/home/f247810/KBQA_WebQSP_improve/data/processed"
BASE_OUT_DIR    = "/home/f247810/KBQA_WebQSP_improve/data/retrieved_paths"
ABLATION_LOG    = "/home/f247810/KBQA_WebQSP_improve/results/retrieval_ablation.json"
CHECKPOINT_DIR  = "/home/f247810/KBQA_WebQSP_improve/data/retrieved_paths/checkpoints"
MAX_LEN         = 128
BATCH_SIZE      = 64
TOP_K_LIST      = [1, 3, 5, 7, 10]
SAVE_EVERY      = 100   # save checkpoint every 100 questions

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# ── checkpoint helpers ─────────────────────────────────────────────────────
def checkpoint_path(split):
    return os.path.join(CHECKPOINT_DIR, f"{split}_checkpoint.json")

def save_checkpoint(split, idx, results_per_k, hits_per_k, hit1_per_k):
    """Save progress to checkpoint file every SAVE_EVERY questions."""
    ckpt = {
        "last_idx"    : idx,
        "hits_per_k"  : hits_per_k,
        "hit1_per_k"  : hit1_per_k,
        "results_per_k": results_per_k
    }
    path = checkpoint_path(split)
    # save to temp first then rename — prevents corrupt checkpoint on crash
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)
    os.replace(tmp_path, path)
    log(f"  [CHECKPOINT] Saved progress at question {idx+1} → {path}")

def load_checkpoint(split):
    """Load checkpoint if exists — returns None if no checkpoint."""
    path = checkpoint_path(split)
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        log(f"  [CHECKPOINT] Resuming {split} from question {ckpt['last_idx']+1}")
        return ckpt
    return None

def delete_checkpoint(split):
    """Delete checkpoint after split completes successfully."""
    path = checkpoint_path(split)
    if os.path.exists(path):
        os.remove(path)
        log(f"  [CHECKPOINT] Deleted checkpoint for {split} — split complete")

# ── setup ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU   : {torch.cuda.get_device_name(0)}")
    log(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── load model ─────────────────────────────────────────────────────────────
log("Loading BERT tokenizer ...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
log("Loading BERT model ...")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.to(device)
model.eval()
log("Model loaded successfully.")

# ── load relation pool ─────────────────────────────────────────────────────
log(f"Loading relation pool ...")
with open(os.path.join(DATA_DIR, "relation_pool.json")) as f:
    relation_pool = json.load(f)
log(f"Relation pool size: {len(relation_pool)}")

# ── create output folders ──────────────────────────────────────────────────
os.makedirs(BASE_OUT_DIR,   exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ABLATION_LOG), exist_ok=True)
for k in TOP_K_LIST:
    os.makedirs(os.path.join(BASE_OUT_DIR, f"top{k}"), exist_ok=True)
log(f"Output folders ready: {[f'top{k}' for k in TOP_K_LIST]}")
log(f"Checkpoint folder  : {CHECKPOINT_DIR}")

# ── scoring function ───────────────────────────────────────────────────────
def score_all_paths(question, paths):
    all_scores = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        enc = tokenizer(
            [question] * len(batch_paths),
            batch_paths,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
                token_type_ids = enc["token_type_ids"].to(device)
            )
        scores = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
        all_scores.extend(scores)
    ranked = sorted(zip(paths, all_scores), key=lambda x: x[1], reverse=True)
    return ranked

# ── process each split ─────────────────────────────────────────────────────
ablation_results = {}

for split in ["train", "dev", "test"]:
    log(f"\n{'='*55}")
    log(f"Processing split: {split}")
    log(f"{'='*55}")

    with open(os.path.join(PROCESSED_DIR, f"{split}.json")) as f:
        data = json.load(f)
    log(f"Loaded {len(data)} questions for {split}")
    total = len(data)

    # ── check for existing checkpoint ──────────────────────────────────────
    ckpt = load_checkpoint(split)
    if ckpt is not None:
        start_idx      = ckpt["last_idx"] + 1
        results_per_k  = {int(k): v for k, v in ckpt["results_per_k"].items()}
        hits_per_k     = {int(k): v for k, v in ckpt["hits_per_k"].items()}
        hit1_per_k     = {int(k): v for k, v in ckpt["hit1_per_k"].items()}
        log(f"  Resuming from question {start_idx} (skipping {start_idx} already done)")
    else:
        start_idx      = 0
        results_per_k  = {k: [] for k in TOP_K_LIST}
        hits_per_k     = {k: 0  for k in TOP_K_LIST}
        hit1_per_k     = {k: 0  for k in TOP_K_LIST}
        log(f"  Starting fresh from question 0")

    # ── score questions ────────────────────────────────────────────────────
    for idx in range(start_idx, total):
        s      = data[idx]
        ranked = score_all_paths(s["question"], relation_pool)
        gold   = " -> ".join(s["inferential_chain"]) \
                 if s["inferential_chain"] else "unknown"

        for k in TOP_K_LIST:
            top_k         = ranked[:k]
            top_path_strs = [p for p, _ in top_k]

            if gold in top_path_strs[:1]: hit1_per_k[k] += 1
            if gold in top_path_strs:     hits_per_k[k] += 1

            results_per_k[k].append({
                "question_id"       : s["question_id"],
                "question"          : s["question"],
                "gold_path"         : gold,
                f"top{k}_paths"     : [
                    {"path": p, "score": round(sc, 4)}
                    for p, sc in top_k
                ],
                "answers"           : s["answers"],
                "sparql"            : s["sparql"],
                "topic_entity_mid"  : s["topic_entity_mid"],
                "topic_entity_name" : s["topic_entity_name"],
                "inferential_chain" : s["inferential_chain"],
                "constraints"       : s["constraints"],
                "all_parses"        : s["all_parses"]
            })

        # ── progress log every 100 ─────────────────────────────────────────
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            log(f"  [{split}] {idx+1}/{total} questions scored ...")

        # ── save checkpoint every 100 questions ────────────────────────────
        if (idx + 1) % SAVE_EVERY == 0 and (idx + 1) < total:
            save_checkpoint(
                split, idx,
                {str(k): v for k, v in results_per_k.items()},
                {str(k): v for k, v in hits_per_k.items()},
                {str(k): v for k, v in hit1_per_k.items()}
            )

    # ── save final results for this split ─────────────────────────────────
    log(f"\nSaving final results for {split} ...")
    for k in TOP_K_LIST:
        out_path = os.path.join(BASE_OUT_DIR, f"top{k}", f"{split}_retrieved.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_per_k[k], f, indent=2, ensure_ascii=False)

        hit1_rate = hit1_per_k[k] / total if total else 0
        hitk_rate = hits_per_k[k] / total if total else 0
        log(f"  Top-{k:2d} | Hit@1={hit1_rate:.4f}  Hit@{k}={hitk_rate:.4f} | saved → top{k}/{split}_retrieved.json")

        key = f"top{k}"
        if key not in ablation_results:
            ablation_results[key] = {}
        ablation_results[key][split] = {
            "hit_at_1"    : round(hit1_rate, 4),
            f"hit_at_{k}" : round(hitk_rate, 4),
            "total"       : total
        }

    # ── delete checkpoint — split done successfully ────────────────────────
    delete_checkpoint(split)

# ── save ablation summary ──────────────────────────────────────────────────
with open(ABLATION_LOG, "w") as f:
    json.dump(ablation_results, f, indent=2)
log(f"\nAblation log saved → {ABLATION_LOG}")

# ── print ablation tables ──────────────────────────────────────────────────
for eval_split in ["dev", "test"]:
    log(f"\n{'='*55}")
    log(f"RETRIEVAL ABLATION — {eval_split.upper()} SET")
    log(f"{'='*55}")
    log(f"{'K':<8} {'Hit@1':<12} {'Hit@K':<12}")
    log("-" * 35)
    for k in TOP_K_LIST:
        key     = f"top{k}"
        metrics = ablation_results[key].get(eval_split, {})
        hit1    = metrics.get("hit_at_1", 0)
        hitk    = metrics.get(f"hit_at_{k}", 0)
        log(f"Top-{k:<4} {hit1:<12.4f} {hitk:<12.4f}")

log(f"\nCheckpoint folder : {CHECKPOINT_DIR}")
log("All checkpoints deleted — run completed successfully.")
log("\nStep 2c COMPLETE.")
