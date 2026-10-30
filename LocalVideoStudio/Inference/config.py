from pathlib import Path

AI_LAB_ROOT = Path("C:/AI-Lab")

MODELS_DIR          = AI_LAB_ROOT / "Models"
RIFE_SCRIPT_DIR     = AI_LAB_ROOT / "ECCV2022-RIFE-main"
RIFE_TRAIN_LOG_DIR  = RIFE_SCRIPT_DIR / "train_log"
INPUT_VIDEO_DIR     = AI_LAB_ROOT / "InputVideo"
OUTPUT_VIDEO_DIR    = AI_LAB_ROOT / "OutputVideo"
VENV_PYTHON         = AI_LAB_ROOT / ".venv" / "Scripts" / "python.exe"

ANIMATEDIFF_MOTION_ADAPTER_DIR = MODELS_DIR / "AnimateDiff-MotionAdapter"
ANIMATEDIFF_BASE_MODEL_DIR     = MODELS_DIR / "AnimateDiff-BaseModel"

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
