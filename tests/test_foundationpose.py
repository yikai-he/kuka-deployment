import sys
from types import ModuleType

import numpy as np

from kuka_deployment.perception.foundationpose import FoundationPoseConfig, FoundationPoseEstimator
from kuka_deployment.perception.interfaces import RGBDFrame


class Float64Mesh:
    """Mirror Trimesh's documented float64 geometry storage contract."""

    def __init__(self):
        self.vertices = np.arange(24, dtype=np.float64).reshape(8, 3)
        self.vertex_normals = np.ones((8, 3), dtype=np.float64)


class FakeFoundationPose:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.diameter = np.float64(0.069282)

    def register(self, **_kwargs):
        return np.eye(4, dtype=np.float64)

    def track_one(self, **_kwargs):
        return np.eye(4, dtype=np.float64)


def test_adapter_separates_trimesh_storage_from_foundationpose_model_dtype(tmp_path, monkeypatch):
    repository = tmp_path / "FoundationPose"
    weights = repository / "weights"
    repository.mkdir()
    weights.mkdir()
    mesh_path = tmp_path / "cube.obj"
    mesh_path.touch()

    trimesh = ModuleType("trimesh")
    trimesh.load = lambda *_args, **_kwargs: Float64Mesh()
    nvdiffrast = ModuleType("nvdiffrast")
    nvdiffrast_torch = ModuleType("nvdiffrast.torch")
    nvdiffrast_torch.RasterizeCudaContext = lambda: object()
    nvdiffrast.torch = nvdiffrast_torch
    estimater = ModuleType("estimater")
    estimater.FoundationPose = FakeFoundationPose
    estimater.PoseRefinePredictor = lambda: object()
    estimater.ScorePredictor = lambda: object()
    for name, module in (("trimesh", trimesh), ("nvdiffrast", nvdiffrast),
                         ("nvdiffrast.torch", nvdiffrast_torch), ("estimater", estimater)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("FOUNDATIONPOSE_WEIGHT_DIR", "previous")

    adapter = FoundationPoseEstimator(FoundationPoseConfig(
        repository=repository, mesh=mesh_path, weights=weights, debug_dir=tmp_path / "debug",
    ))
    supplied = FakeFoundationPose.last_kwargs
    assert supplied["mesh"].vertices.dtype == np.float64
    assert supplied["mesh"].vertex_normals.dtype == np.float64
    assert supplied["model_pts"].dtype == np.float32
    assert supplied["model_normals"].dtype == np.float32
    assert supplied["model_pts"].flags.c_contiguous
    assert supplied["model_normals"].flags.c_contiguous
    assert adapter.dtype_diagnostics == {
        "register_calls": 0, "track_calls": 0,
        "register_input_dtypes": [], "track_input_dtypes": [], "output_dtypes": [],
        "mesh_vertices_dtype": "float64", "mesh_vertex_normals_dtype": "float64",
        "model_pts_dtype": "float32", "model_normals_dtype": "float32",
        "diameter_dtype": "float32",
    }

    frame = RGBDFrame(
        np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32),
        np.eye(3, dtype=np.float32), 0, None, "camera",
    )
    assert adapter.register(frame, np.ones((2, 2), bool)).dtype == np.float32
    assert adapter.track(frame).dtype == np.float32
    assert adapter.dtype_diagnostics["output_dtypes"] == ["float32", "float32"]
