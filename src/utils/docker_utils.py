"""Path/environment helpers that behave the same locally and inside the container."""
from pathlib import Path

from src.utils.config import DATA_DIR, MODELS_DIR


def ensure_runtime_dirs() -> None:
    """Create the data/models directories if they don't exist yet (fresh clone/volume)."""
    for d in (DATA_DIR, MODELS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def model_artifacts_present() -> bool:
    from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH

    return MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
