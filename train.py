"""
train.py

Fine-tune multilingual-e5 model, and evaluated using tag-based IDF similarity.
Output is saved to: output/attempt_NN/model and output/attempt_NN/log.json
    model contains saved SentenceTransformer
    log.json contains training hyperparams and per-step loss
"""

from datetime import datetime

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback, TrainerControl, TrainerState

from data_for_model import get_data_for_model
from output_manager import get_next_attempt_dir, list_attempts, save_log
from tag_similarity import build_pairs, get_tag_counts

BASE_MODEL   = "intfloat/multilingual-e5-small"
EPOCHS       = 5
BATCH_SIZE   = 8
LR           = 2e-5
WARMUP_RATIO = 0.1

def post_to_text(post: dict) -> str:
    comments = " ".join(post.get("comments", []))
    return f"{post['title']}. {post['body']} {comments}".strip()

class StepLogger(TrainerCallback):
    def __init__(self):
        self.entries: list[dict] = []

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if logs and "loss" in logs:
            entry = {
                "step":  state.global_step,
                "epoch": round(state.epoch, 3) if state.epoch is not None else None,
                "loss":  round(logs["loss"], 6),
            }
            self.entries.append(entry)
            print(f"  step {entry['step']:>4} | epoch {entry['epoch']} | loss {entry['loss']:.6f}")

def train():
    #prepare data and dir
    prev = list_attempts()
    if prev:
        print("Existing attempts:")
        for p in prev:
            print(f"  {p.name}")
    else:
        print("No previous attempts found.")

    attempt_dir = get_next_attempt_dir()
    model_dir   = attempt_dir / "model"

    print(f"\n{'='*52}")
    print(f"  Starting: {attempt_dir.name}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*52}\n")

    data       = get_data_for_model()
    tag_counts = get_tag_counts(data)
    n_posts    = len(data)
    pairs      = build_pairs(data, tag_counts)

    print(f"Posts : {n_posts}")
    print(f"Tags  : {tag_counts}\n")

    texts   = [post_to_text(p) for p in data]
    dataset = Dataset.from_dict({
        "sentence1": [texts[a] for a, _, _ in pairs],
        "sentence2": [texts[b] for _, b, _ in pairs],
        "label":     [float(s) for _, _, s in pairs],
    })

    score_vals = [s for _, _, s in pairs]
    print(f"Training pairs : {len(pairs)}")
    print(f"Score range    : [{min(score_vals):.3f}, {max(score_vals):.3f}]")
    print(f"Non-zero pairs : {sum(1 for s in score_vals if s > 0)}\n")

    #model setting
    print(f"Loading model  : {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    loss  = CosineSimilarityLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(attempt_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        report_to="none",
    )

    logger  = StepLogger()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
        callbacks=[logger],
    )
    #train
    print("Training...\n")
    trainer.train()

    model.save(str(model_dir))
    print(f"\nModel saved : {model_dir}")

    log_data = {
        "attempt":        attempt_dir.name,
        "timestamp":      datetime.now().isoformat(),
        "base_model":     BASE_MODEL,
        "epochs":         EPOCHS,
        "batch_size":     BATCH_SIZE,
        "learning_rate":  LR,
        "warmup_ratio":   WARMUP_RATIO,
        "n_posts":        n_posts,
        "n_pairs":        len(pairs),
        "tag_counts":     tag_counts,
        "training_steps": logger.entries,
    }
    log_path = save_log(attempt_dir, log_data)
    print(f"Log saved   : {log_path}\n")


if __name__ == "__main__":
    train()
