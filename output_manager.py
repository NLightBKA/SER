import json
import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"

#find all the attempt_NN folders in output folder
def list_attempts() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        (d for d in OUTPUT_DIR.iterdir()
         if d.is_dir() and re.match(r"attempt_\d+$", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

#name the next attempt_NN folder and turn dir
def get_next_attempt_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = [
        int(re.search(r"\d+", d.name).group())
        for d in list_attempts()
    ]
    next_n = int(max(existing) + 1) if existing else 1
    path = OUTPUT_DIR / f"attempt_{next_n:02d}"
    path.mkdir(parents=True)
    return path


def save_log(attempt_dir: Path, log_data: dict) -> Path:
    log_path = attempt_dir / "log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    return log_path

