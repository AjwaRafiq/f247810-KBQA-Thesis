import json
import os
import random

# ── paths ──────────────────────────────────────────────────────────────────
TRAIN_FILE  = "/home/f247810/KBQA_WebQSP_improve/data/processed/train.json"
DEV_FILE    = "/home/f247810/KBQA_WebQSP_improve/data/processed/dev.json"
OUT_DIR     = "/home/f247810/KBQA_WebQSP_improve/data/bert_input"
SEED        = 42

random.seed(SEED)

# ── collect all unique relation paths from train ───────────────────────────
def chain_to_str(chain):
    if not chain:
        return "unknown"
    return " -> ".join(chain)

def load_data(path):
    with open(path) as f:
        return json.load(f)

print("Loading processed data ...")
train_data = load_data(TRAIN_FILE)
dev_data   = load_data(DEV_FILE)

# collect all unique relation paths seen in training
all_chains = set()
for s in train_data:
    c = chain_to_str(s["inferential_chain"])
    all_chains.add(c)
# also add from dev so we can evaluate
for s in dev_data:
    c = chain_to_str(s["inferential_chain"])
    all_chains.add(c)

all_chains = sorted(list(all_chains))
print(f"Total unique relation paths: {len(all_chains)}")

# ── build training pairs ───────────────────────────────────────────────────
# For each question:
#   positive = its ground truth inferential chain  → label 1
#   negatives = random other chains from pool       → label 0
# Hard negatives = chains that share at least one relation with positive

NEG_PER_POS = 4  # 1 positive : 4 negatives per question

def get_hard_negatives(pos_chain, all_chains_list, n=2):
    pos_rels = set(pos_chain)
    hard = [c for c in all_chains_list
            if c != pos_chain
            and any(r in c for r in pos_rels)]
    random.shuffle(hard)
    return hard[:n]

def get_random_negatives(pos_chain, all_chains_list, n=2):
    pool = [c for c in all_chains_list if c != pos_chain]
    random.shuffle(pool)
    return pool[:n]

def build_pairs(data, split_name):
    pairs = []
    for s in data:
        question   = s["question"]
        pos_chain  = chain_to_str(s["inferential_chain"])

        # positive sample
        pairs.append({
            "question"      : question,
            "relation_path" : pos_chain,
            "label"         : 1,
            "question_id"   : s["question_id"]
        })

        # hard negatives (2)
        hard_negs = get_hard_negatives(pos_chain, all_chains, n=2)
        for neg in hard_negs:
            pairs.append({
                "question"      : question,
                "relation_path" : neg,
                "label"         : 0,
                "question_id"   : s["question_id"]
            })

        # random negatives (2)
        rand_negs = get_random_negatives(pos_chain, all_chains, n=2)
        for neg in rand_negs:
            pairs.append({
                "question"      : question,
                "relation_path" : neg,
                "label"         : 0,
                "question_id"   : s["question_id"]
            })

    random.shuffle(pairs)
    print(f"  {split_name}: {len(pairs)} pairs "
          f"(pos={sum(1 for p in pairs if p['label']==1)}, "
          f"neg={sum(1 for p in pairs if p['label']==0)})")
    return pairs

print("\nBuilding training pairs ...")
train_pairs = build_pairs(train_data, "train")
print("Building dev pairs ...")
dev_pairs   = build_pairs(dev_data,   "dev")

# ── save ───────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for name, pairs in [("train", train_pairs), ("dev", dev_pairs)]:
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {name} → {out_path}")

# save relation path pool for retrieval at inference time
pool_path = os.path.join(OUT_DIR, "relation_pool.json")
with open(pool_path, "w") as f:
    json.dump(all_chains, f, indent=2)
print(f"Saved relation pool ({len(all_chains)} paths) → {pool_path}")

# ── summary ────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("BERT DATA PREPARATION SUMMARY")
print("=" * 50)
print(f"Unique relation paths  : {len(all_chains)}")
print(f"Train pairs            : {len(train_pairs)}")
print(f"Dev   pairs            : {len(dev_pairs)}")
print(f"Negatives per question : {NEG_PER_POS} (2 hard + 2 random)")
print(f"Output dir             : {OUT_DIR}")
print("=" * 50)
print("\nStep 2a COMPLETE — BERT data prepared.")
