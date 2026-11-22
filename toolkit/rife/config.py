from dataclasses import dataclass


@dataclass
class InterpolationConfig:
    """Controls RIFE frame interpolation behaviour."""
    exp: int = 1          # 2^exp multiplier: 1 = 2× (30→60 fps), 2 = 4× (30→120 fps)
    scale: float = 1.0    # optical-flow resolution scale; try 0.5 for 4K inputs
    fp16: bool = True     # half-precision on Tensor Core GPUs (RTX series)
