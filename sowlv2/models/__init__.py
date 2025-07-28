"""Model wrappers for SOWLv2 (OWLv2, SAM2, and EdgeTAM)."""
from .owl import OWLV2Wrapper

# Conditional imports to avoid dependency issues
try:
    from .sam2_wrapper import SAM2Wrapper
except ImportError:
    SAM2Wrapper = None

try:
    from .edgetam_wrapper import EdgeTAMWrapper
except ImportError:
    EdgeTAMWrapper = None

# Model factory should always be available since it handles fallbacks
from .model_factory import SegmentationModelFactory

__all__ = [
    'OWLV2Wrapper',
    'SAM2Wrapper',
    'EdgeTAMWrapper',
    'SegmentationModelFactory'
]
