import numpy as np

from cv_emulator_pipeline.core import extract_geometry_features, target_point_from_box


def test_target_point_from_box():
    x, y = target_point_from_box(np.array([10, 20, 30, 60]), vertical_anchor=0.5)
    assert x == 20
    assert y == 40


def test_geometry_features():
    features = extract_geometry_features(np.array([0, 0, 50, 100]), (200, 100, 3))
    assert features["w_norm"] == 0.5
    assert features["h_norm"] == 0.5
