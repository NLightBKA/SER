"""
train.py

Fine-tune multilingual-e5 model, and evaluated using tag-based IDF similarity.
Output is saved to: output/attempt_NN/model and output/attempt_NN/log.json
    model contains saved SentenceTransformer
    log.json contains training hyperparams and per-step loss
"""

from datetime import datetime
from pathlib import Path

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
    def __init__(self, attempt_dir: Path, meta_info: dict):
        self.attempt_dir = attempt_dir
        self.meta_info   = meta_info
        self.entries: list[dict] = []
        # Khởi tạo file log.json ngay lập tức với metadata
        self.flush()

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if not logs:
            return

        entry = {
            "step":  state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch is not None else None,
        }
        if "loss" in logs:
            entry["loss"] = round(logs["loss"], 6)
        if "eval_loss" in logs:
            entry["eval_loss"] = round(logs["eval_loss"], 6)
        if "learning_rate" in logs:
            entry["learning_rate"] = logs["learning_rate"]

        self.entries.append(entry)

        # In thông tin ra terminal
        display_parts = [f"  step {entry['step']:>4}"]
        if entry.get("epoch") is not None:
            display_parts.append(f"epoch {entry['epoch']}")
        if "loss" in entry:
            display_parts.append(f"loss {entry['loss']:.6f}")
        if "eval_loss" in entry:
            display_parts.append(f"eval_loss {entry['eval_loss']:.6f}")
        print(" | ".join(display_parts))

        # Cập nhật thời gian thực vào file log.json
        self.flush()

    def flush(self):
        log_data = {
            **self.meta_info,
            "training_steps": self.entries,
        }
        save_log(self.attempt_dir, log_data)


def train():
    #prepare dir
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

    #prepare data
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

    #initialize model and loss
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
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    meta_info = {
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
    }

    logger = StepLogger(attempt_dir, meta_info)
    print(f"Log initialized: {attempt_dir / 'log.json'}\n")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
        callbacks=[logger],
    )

    #train
    try:
        print("Training...\n")
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining Interuppted. Saving current state...")
    finally:
        model.save(str(model_dir))
        print(f"\nModel saved : {model_dir}")
        logger.flush()
        print(f"Log saved   : {attempt_dir / 'log.json'}\n")


if __name__ == "__main__":
    train()
