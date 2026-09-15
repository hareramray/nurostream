"""neurostream - run transformers larger than memory.

Model size is bounded by disk capacity, not RAM or VRAM. Weights stream from
NVMe on demand at neuron granularity, under a hard memory ceiling that holds
regardless of model size.
"""
from .api import NeuroStream, GenerationStats

__version__ = "0.2.0"
__all__ = ["NeuroStream", "GenerationStats"]
