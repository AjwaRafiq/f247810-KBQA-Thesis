import json
import os
import re
import sys

RETRIEVED_DIR = "/home/f247810/KBQA_WebQSP_improve/data/retrieved_paths/top5"
OUT_DIR = "/home/f247810/KBQA_WebQSP_improve/data/llama_input"
TOP_K = 5


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def question_to_statement(question, topic_entity):
    q = question.strip().lower()
    q = re.sub(r'\?$', '', q).strip()

    entity = topic_entity.strip() if topic_entity else ""
    entity_lower = entity.lower()

    if entity_lower and entity_lower in q:
        q = q.replace(entity_lower, entity).strip()

    if re.match(r'^what (is|are|was|were) ', q):
        stmt = re.sub(r'^what (is|are|was|were) ', '', q)
        stmt = re.sub(r'^the ', '', stmt)
        return f"The {stmt} is *placeholder*"

    if re.match(r'^what did ', q):
        stmt = re.sub(r'^what did ', '', q)
        return f"{stmt} *placeholder*"

    if re.match(r'^who (is|was|were) ', q):
        stmt = re.sub(r'^who (is|was|were) ', '', q)
        return f"The person {stmt} is *placeholder*"

    if re.match(r'^who (did|does|do) ', q):
        stmt = re.sub(r'^who (did|does|do) ', '', q)
        return f"{stmt} *placeholder*"

    if re.match(r'^where (is|was|are|were) ', q):
        stmt = re.sub(r'^where (is|was|are|were) ', '', q)
        return f"{stmt} is located at *placeholder*"

    if re.match(r'^when (did|was|is|were) ', q):
        stmt = re.sub(r'^when (did|was|is|were) ', '', q)
        return f"{stmt} happened at *placeholder*"

    if re.match(r'^which ', q):
        stmt = re.sub(r'^which ', '', q)
        return f"The {stmt} is *placeholder*"

    if re.match(r'^how (many|much) ', q):
        stmt = re.sub(r'^how (many|much) ', '', q)
        return f"The number of {stmt} is *placeholder*"

    if re.match(r'^what (type|kind|sort) ', q):
        stmt = re.sub(r'^what (type|kind|sort) (of )?', '', q)
        return f"The type of {stmt} is *placeholder*"

    if re.match(r'^what ', q):
        stmt = re.sub(r'^what ', '', q)
        return f"The {stmt} is *placeholder*"

    return f"{q} is *placeholder*"


def build_proposition(idx, question, path_str, topic_entity, topic_mid):
    relations = path_str.split(" -> ") if path_str != "unknown" else []

    if topic_entity and relations:
        chain_parts = [topic_entity]
        chain_parts.extend(relations)
        chain_parts.append("[answer]")
        premise = " -> ".join(chain_parts)
    elif relations:
        premise = " -> ".join(relations) + " -> [answer]"
    else:
        premise = "unknown -> [answer]"

    conclusion = question_to_statement(question, topic_entity)

    return {
        "index": idx,
        "premise": premise,
        "conclusion": conclusion,
        "path_str": path_str,
    }


def build_instruction(sample, propositions):
    prop_text = ""
    for p in propositions:
        prop_text += (
            f"Proposition {p['index']}: "
            f"The premise is: {p['premise']}. "
            f"Its conclusion is: {p['conclusion']}\n"
        )

    return (
        f"Given a question: ({sample['question']})\n\n"
        f"And a series of Propositions:\n"
        f"{prop_text}\n"
        f"Task 1: Generate logical forms just based on the question.\n"
        f"Task 2: Verify which deductive reasoning is correct for the "
        f"given question in a deductive manner. If it exists, return the "
        f"correct number of deductive reasoning. If it does not exist, "
        f"return 'no'."
    )


def find_correct_index(propositions, gold_path):
    for p in propositions:
        if p["path_str"] == gold_path:
            return str(p["index"])
    return "no"


def build_sample(raw):
    question = raw["question"]
    gold_path = raw["gold_path"]
    sparql = raw["sparql"]
    topic_entity = raw.get("topic_entity_name", "")
    topic_mid = raw.get("topic_entity_mid", "")
    answers = raw.get("answers", [])
    top5_paths = raw.get("top5_paths", [])

    propositions = []
    for i, path_info in enumerate(top5_paths):
        propositions.append(
            build_proposition(
                idx=i,
                question=question,
                path_str=path_info["path"],
                topic_entity=topic_entity,
                topic_mid=topic_mid,
            )
        )

    instruction = build_instruction(raw, propositions)
    correct_index = find_correct_index(propositions, gold_path)
    lf = sparql.strip() if sparql else "unknown"

    output = (
        f"Logical form: {lf}\n"
        f"Path selection: deductive reasoning {correct_index}"
    )

    return {
        "question_id": raw["question_id"],
        "question": question,
        "instruction": instruction,
        "output": output,
        "task1_lf": lf,
        "task2_index": correct_index,
        "gold_path": gold_path,
        "answers": answers,
        "topic_entity": topic_entity,
        "propositions": propositions,
    }


def process_split(split):
    in_path = os.path.join(RETRIEVED_DIR, f"{split}_retrieved.json")
    out_path = os.path.join(OUT_DIR, f"{split}_instructions.json")

    log(f"\nProcessing {split} ...")

    with open(in_path) as f:
        data = json.load(f)

    log(f"  Loaded {len(data)} samples")

    results = []
    task2_no_count = 0
    empty_lf_count = 0

    for s in data:
        sample = build_sample(s)
        results.append(sample)

        if sample["task2_index"] == "no":
            task2_no_count += 1

        if not sample["task1_lf"] or sample["task1_lf"] == "unknown":
            empty_lf_count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"  Total samples          : {len(results)}")
    log(f"  Task2 'no' (gold not in top5): {task2_no_count} ({task2_no_count/len(results)*100:.1f}%)")
    log(f"  Empty LF               : {empty_lf_count}")
    log(f"  Saved → {out_path}")

    ex = results[0]

    log("\n  === EXAMPLE ===")
    log(f"  Question   : {ex['question']}")
    log("  Instruction preview:")

    for line in ex["instruction"].split("\n")[:8]:
        log(f"    {line}")

    log(f"  Output     : {ex['output'][:100]}...")
    log(f"  Task2 index: {ex['task2_index']}")

    return results


os.makedirs(OUT_DIR, exist_ok=True)

log("=" * 55)
log("STEP 3: Llama Instruction Data Preparation")
log("=" * 55)
log(f"Using Top-{TOP_K} retrieved paths")
log("Rule-based question-to-statement converter")

log("\n── Converter test ──────────────────────────────────────")

test_cases = [
    ("what is the name of justin bieber brother", "Justin Bieber"),
    ("who does joakim noah play for", "Joakim Noah"),
    ("what kind of money to take to bahamas", "Bahamas"),
    ("where was barack obama born", "Barack Obama"),
    ("when did world war 2 end", "World War 2"),
    ("how many seasons does breaking bad have", "Breaking Bad"),
    ("which country is paris the capital of", "Paris"),
]

for q, e in test_cases:
    stmt = question_to_statement(q, e)
    log(f"  Q: {q}")
    log(f"  S: {stmt}")
    log("")

train_data = process_split("train")
dev_data = process_split("dev")

train_ids = set(s["question_id"] for s in train_data)
dev_ids = set(s["question_id"] for s in dev_data)

assert len(train_ids & dev_ids) == 0, "LEAKAGE in llama data!"

log("\nLeakage check PASSED")

log("\n" + "=" * 55)
log("STEP 3 COMPLETE")
log("=" * 55)
log(f"Train instructions : {len(train_data)}")
log(f"Dev   instructions : {len(dev_data)}")
log(f"Output dir         : {OUT_DIR}")