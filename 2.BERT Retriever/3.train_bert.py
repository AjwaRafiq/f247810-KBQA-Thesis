import json
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW

# ── config ─────────────────────────────────────────────────────────────────
BERT_MODEL_PATH = "/home/f247810/BertT5-LLaMA-KBQA/models/bert_ranker/bert-base-uncased"
DATA_DIR        = "/home/f247810/KBQA_WebQSP_improve/data/bert_input"
OUT_DIR         = "/home/f247810/KBQA_WebQSP_improve/models/BERT_retriever"
LOG_FILE        = "/home/f247810/KBQA_WebQSP_improve/results/bert_training_log.json"
RESULT_FILE     = "/home/f247810/KBQA_WebQSP_improve/results/bert_training_analysis.txt"

MAX_LEN             = 128
BATCH_SIZE          = 32
EPOCHS              = 15       
LR                  = 2e-5
WARMUP              = 0.1
SEED                = 42
LAMBDA_CON          = 1.0
WEIGHT_DECAY        = 0.02     
CLASSIFIER_DROPOUT  = 0.1     
EARLY_STOP_PATIENCE = 3       
EARLY_STOP_METRIC   = "f1"    

# ── reproducibility ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── dataset ────────────────────────────────────────────────────────────────
class BertRetrieverDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p   = self.pairs[idx]
        enc = self.tokenizer(
            p["question"],
            p["relation_path"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "token_type_ids" : enc["token_type_ids"].squeeze(0),
            "label"          : torch.tensor(p["label"], dtype=torch.long)
        }

# ── confidence advantage loss ──────────────────────────────────────────────
def confidence_advantage_loss(logits, labels, lambda_val=1.0):
    pos_scores = logits[labels == 1, 1]
    neg_scores = logits[labels == 0, 1]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=logits.device)
    pos_mean = pos_scores.mean()
    neg_mean = neg_scores.mean()
    loss = torch.log(1 + torch.exp(lambda_val * (neg_mean - pos_mean)))
    return loss

# ── metrics ────────────────────────────────────────────────────────────────
def compute_metrics(logits_list, labels_list, loss_list=None):
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds  = logits.argmax(dim=-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    metrics = {
        "acc"      : round(acc, 4),
        "f1"       : round(f1, 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4)
    }

      if loss_list is not None:
        avg_val_loss = sum(loss_list) / len(loss_list)
        metrics["val_loss"] = round(avg_val_loss, 4)

    return metrics

# ── load data ──────────────────────────────────────────────────────────────
print("\nLoading data ...")
with open(os.path.join(DATA_DIR, "train.json")) as f:
    train_pairs = json.load(f)
with open(os.path.join(DATA_DIR, "dev.json")) as f:
    dev_pairs = json.load(f)

print(f"Train pairs : {len(train_pairs)}")
print(f"Dev   pairs : {len(dev_pairs)}")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
train_ds  = BertRetrieverDataset(train_pairs, tokenizer, MAX_LEN)
dev_ds    = BertRetrieverDataset(dev_pairs,   tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── model with classifier dropout ─────────────────────────────────────────
# FIX 2: add classifier dropout to BERT config
print("\nLoading BERT model with classifier dropout ...")
config = BertConfig.from_pretrained(BERT_MODEL_PATH)
config.num_labels          = 2
config.classifier_dropout  = CLASSIFIER_DROPOUT   
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH,
    config=config
)
model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")
print(f"Classifier dropout applied : {CLASSIFIER_DROPOUT}")
print(f"Weight decay               : {WEIGHT_DECAY}")
print(f"Early stopping patience    : {EARLY_STOP_PATIENCE} epochs")
print(f"Early stopping metric      : {EARLY_STOP_METRIC}")

# ── optimizer and scheduler ────────────────────────────────────────────────
optimizer     = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP)
scheduler     = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

ce_loss_fn = torch.nn.CrossEntropyLoss()

# ── training loop ──────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

best_metric        = 0.0
best_epoch         = 0
no_improve_count   = 0
training_log       = []
stopped_early      = False

print("\nStarting training ...")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    # ── train ──────────────────────────────────────────────────────────────
    model.train()
    total_train_loss = 0.0
    steps            = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs  = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        logits   = outputs.logits
        loss_ce  = ce_loss_fn(logits, labels)
        loss_con = confidence_advantage_loss(logits, labels, LAMBDA_CON)
        loss     = loss_ce + loss_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        steps            += 1

    avg_train_loss = total_train_loss / steps

    # ── evaluate on dev ────────────────────────────────────────────────────
    model.eval()
    all_logits    = []
    all_labels    = []
    all_val_losses = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs  = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )
            logits   = outputs.logits

         
            loss_ce  = ce_loss_fn(logits, labels)
            loss_con = confidence_advantage_loss(logits, labels, LAMBDA_CON)
            val_loss = (loss_ce + loss_con).item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_val_losses.append(val_loss)

    metrics = compute_metrics(all_logits, all_labels, all_val_losses)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"val_loss={metrics['val_loss']:.4f} | "
        f"acc={metrics['acc']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"p={metrics['precision']:.4f} | "
        f"r={metrics['recall']:.4f}"
    )

    log_entry = {
        "epoch"      : epoch,
        "train_loss" : round(avg_train_loss, 4),
        **metrics
    }
    training_log.append(log_entry)

    # ── save best model ────────────────────────────────────────────────────
    current_metric = metrics[EARLY_STOP_METRIC]
    if current_metric > best_metric:
        best_metric      = current_metric
        best_epoch       = epoch
        no_improve_count = 0
        model.save_pretrained(os.path.join(OUT_DIR, "best"))
        tokenizer.save_pretrained(os.path.join(OUT_DIR, "best"))
        print(f"   Best model saved  (f1={best_metric:.4f}  val_loss={metrics['val_loss']:.4f})")
    else:
        no_improve_count += 1
        print(f"  No improvement for {no_improve_count}/{EARLY_STOP_PATIENCE} epochs")

    # early stopping check
    if no_improve_count >= EARLY_STOP_PATIENCE:
        print(f"\n  EARLY STOPPING at epoch {epoch} ")
        print(f"  No improvement in {EARLY_STOP_METRIC} for {EARLY_STOP_PATIENCE} epochs")
        print(f"  Best epoch was {best_epoch} with {EARLY_STOP_METRIC}={best_metric:.4f}")
        stopped_early = True
        break

# ── save final checkpoint ──────────────────────────────────────────────────
model.save_pretrained(os.path.join(OUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUT_DIR, "final"))

# ── save training log ──────────────────────────────────────────────────────
with open(LOG_FILE, "w") as f:
    json.dump(training_log, f, indent=2)
print(f"\nTraining log saved → {LOG_FILE}")

# ── full analysis report ───────────────────────────────────────────────────
best_log  = max(training_log, key=lambda x: x["f1"])
lines = []
lines.append("=" * 70)
lines.append("BERT RETRIEVER TRAINING ANALYSIS")
lines.append("=" * 70)
lines.append("")
lines.append("── Config ─────────────────────────────────────────────────────────")
lines.append(f"  Epochs planned      : {EPOCHS}")
lines.append(f"  Epochs trained      : {len(training_log)}")
lines.append(f"  Early stopped       : {stopped_early}")
lines.append(f"  Classifier dropout  : {CLASSIFIER_DROPOUT}")
lines.append(f"  Weight decay        : {WEIGHT_DECAY}")
lines.append(f"  Early stop patience : {EARLY_STOP_PATIENCE}")
lines.append(f"  Early stop metric   : {EARLY_STOP_METRIC}")
lines.append("")
lines.append("── Full Training Log ───────────────────────────────────────────────")
lines.append(f"{'Epoch':<8}{'TrainLoss':<12}{'ValLoss':<12}{'Acc':<10}{'F1':<10}{'Precision':<12}{'Recall':<10}")
lines.append("-" * 70)
for e in training_log:
    marker = " ← BEST" if e["epoch"] == best_log["epoch"] else ""
    lines.append(
        f"{e['epoch']:<8}"
        f"{e['train_loss']:<12}"
        f"{e.get('val_loss','N/A'):<12}"
        f"{e['acc']:<10}"
        f"{e['f1']:<10}"
        f"{e['precision']:<12}"
        f"{e['recall']:<10}"
        f"{marker}"
    )
lines.append("")
lines.append("── Best Epoch Per Metric ───────────────────────────────────────────")
best_f1  = max(training_log, key=lambda x: x["f1"])
best_acc = max(training_log, key=lambda x: x["acc"])
best_p   = max(training_log, key=lambda x: x["precision"])
best_r   = max(training_log, key=lambda x: x["recall"])
best_vl  = min(training_log, key=lambda x: x.get("val_loss", 999))
lines.append(f"  Best F1        : Epoch {best_f1['epoch']}  → F1={best_f1['f1']}")
lines.append(f"  Best Accuracy  : Epoch {best_acc['epoch']}  → Acc={best_acc['acc']}")
lines.append(f"  Best Precision : Epoch {best_p['epoch']}  → P={best_p['precision']}")
lines.append(f"  Best Recall    : Epoch {best_r['epoch']}  → R={best_r['recall']}")
lines.append(f"  Best Val Loss  : Epoch {best_vl['epoch']}  → ValLoss={best_vl.get('val_loss','N/A')}")
lines.append("")
lines.append("── Overfitting Analysis ────────────────────────────────────────────")

# check train loss vs val loss gap
if len(training_log) >= 2:
    first      = training_log[0]
    last       = training_log[-1]
    best_e     = training_log[best_log["epoch"] - 1]
    loss_gap   = last["train_loss"] - last.get("val_loss", 0)
    f1_dropped = best_log["f1"] > last["f1"]

    lines.append(f"  Train loss (epoch 1)     : {first['train_loss']}")
    lines.append(f"  Train loss (last epoch)  : {last['train_loss']}")
    lines.append(f"  Val   loss (last epoch)  : {last.get('val_loss', 'N/A')}")
    lines.append(f"  Train/Val loss gap       : {abs(loss_gap):.4f}")
    lines.append(f"  F1 dropped after best    : {f1_dropped}")
    lines.append("")

    if f1_dropped:
        lines.append("  OVERFITTING DETECTED — F1 dropped after best epoch")
        lines.append(f"  USE EPOCH {best_log['epoch']} checkpoint only")
    elif abs(loss_gap) > 0.05:
        lines.append("  MILD OVERFITTING — train/val loss gap is large")
        lines.append(f"  USE EPOCH {best_log['epoch']} checkpoint")
    else:
        lines.append("  NO overfitting — train and val loss aligned")
        lines.append(" Model converged cleanly")

lines.append("")
lines.append("── Early Stopping ──────────────────────────────────────────────────")
if stopped_early:
    lines.append(f"  Triggered at epoch {len(training_log)}")
    lines.append(f"  Best epoch was {best_epoch}")
    lines.append(f"  Saved {EARLY_STOP_PATIENCE} wasted epochs")
else:
    lines.append("  Not triggered — model kept improving until max epochs")

lines.append("")
lines.append("── Recommendation ──────────────────────────────────────────────────")
lines.append(f"  USE MODEL      : models/BERT_retriever/best/")
lines.append(f"  BEST EPOCH     : {best_log['epoch']}")
lines.append(f"  BEST F1        : {best_log['f1']}")
lines.append(f"  BEST ACC       : {best_log['acc']}")
lines.append(f"  BEST PRECISION : {best_log['precision']}")
lines.append(f"  BEST RECALL    : {best_log['recall']}")
lines.append("=" * 70)

for l in lines:
    print(l)

with open(RESULT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nAnalysis report saved → {RESULT_FILE}")
print("\nStep 2b COMPLETE.")
