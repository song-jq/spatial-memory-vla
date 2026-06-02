from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "3d_encoder.py"
_SPEC = spec_from_file_location("spatial_memory_vla_3d_encoder", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

DEFAULT_3D_ENCODER_CHECKPOINT = _MODULE.DEFAULT_3D_ENCODER_CHECKPOINT
STN3d = _MODULE.STN3d
STNkd = _MODULE.STNkd
PointNetfeat = _MODULE.PointNetfeat
DepthPerceptionProjector = _MODULE.DepthPerceptionProjector
depth_to_point_cloud = _MODULE.depth_to_point_cloud
coerce_depth_tensor = _MODULE.coerce_depth_tensor
process_depth_features = _MODULE.process_depth_features
process_depth_perception_tokens = _MODULE.process_depth_perception_tokens
append_depth_tokens = _MODULE.append_depth_tokens
get_depth_projectors_for_checkpoint = _MODULE.get_depth_projectors_for_checkpoint

__all__ = [
    "DEFAULT_3D_ENCODER_CHECKPOINT",
    "STN3d",
    "STNkd",
    "PointNetfeat",
    "DepthPerceptionProjector",
    "depth_to_point_cloud",
    "coerce_depth_tensor",
    "process_depth_features",
    "process_depth_perception_tokens",
    "append_depth_tokens",
    "get_depth_projectors_for_checkpoint",
]
