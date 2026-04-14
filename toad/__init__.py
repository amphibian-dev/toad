from .merge import merge, DTMerge, ChiMerge, StepMerge, QuantileMerge, KMeansMerge

from .detector import detect
from .metrics import KS, KS_bucket, F1
from .stats import quality, IV, VIF, WOE, entropy, entropy_cond, gini, gini_cond
from .transform import Combiner, WOETransformer
from .selection import select
from .scorecard import ScoreCard
from .utils import Progress, performance
from .version import __version__

# Expose rust module at package level
try:
    import toad.rust
    rust = toad.rust
except ImportError:
    rust = None

VERSION = __version__