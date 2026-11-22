"""
LocalVideoStudio runtime configuration.

All paths and defaults are defined here so the pipeline and CLI never
hard-code anything.
"""

from pathlib import Path

AI_LAB_ROOT     = Path(__file__).parent.parent.parent   # C:\AI-Lab
OUTPUT_VIDEO_DIR = AI_LAB_ROOT / "OutputVideo"
INPUT_VIDEO_DIR  = AI_LAB_ROOT / "InputVideo"

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
