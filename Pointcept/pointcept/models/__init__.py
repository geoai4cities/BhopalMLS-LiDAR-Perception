from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier
from .modules import PointModule, PointModel

# Backbones
from .sparse_unet import *
from .point_transformer import *
from .point_transformer_v2 import *
from .point_transformer_v3 import *
from .stratified_transformer import *
from .spvcnn import *
from .octformer import *
from .oacnns import *

# from .swin3d import *

# Semantic Segmentation
from .context_aware_classifier import *

# Instance Segmentation
from .point_group import *
from .sgiformer import *

# Pretraining
from .masked_scene_contrast import *
from .point_prompt_training import *
from .sonata import *
from .concerto import *

# KPConvX (Apple ml-kpconvx, Pointcept port). Needs the compiled C++ wrappers; see
# setup_venv.sh --with-kpconvx. Left out with a warning when they are absent so the other
# models keep working.
try:
    from .kpconvx import *  # noqa: F401,F403
except ImportError as _e:  # pragma: no cover
    import warnings as _w
    _w.warn(f"KPConvX unavailable ({_e}); run setup_venv.sh --with-kpconvx to build its C++ wrappers")
