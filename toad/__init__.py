__version_info__ = (0, 0, 33)
__version__ = '.'.join(str(x) for x in __version_info__)
VERSION = __version__

from .merge import merge, DTMerge, ChiMerge, StepMerge, QuantileMerge, KMeansMerge
from .detector import detect
from .stats import quality, IV, WOE, KS, KS_bucket, entropy, entropy_cond, gini, gini_cond
from .selection import select
from .scorecard import ScoreCard
