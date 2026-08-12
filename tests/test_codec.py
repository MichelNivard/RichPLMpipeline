import numpy as np

from proteinloss_pipeline.codec import decode_mi_target, distance_log_target, encode_ca_coords, encode_mi_target
from proteinloss_pipeline.targets import validate_pair_target


def test_dense_mi_codec_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(31, 31)).astype(np.float32)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    decoded = decode_mi_target(encode_mi_target(matrix))
    validate_pair_target(decoded)
    assert np.sqrt(np.mean((decoded - matrix) ** 2)) < 0.8


def test_ca_distance_target_is_symmetric_with_zero_diagonal():
    coordinates = np.arange(45, dtype=np.float32).reshape(15, 3) / 5
    distance = distance_log_target(encode_ca_coords(coordinates))
    validate_pair_target(distance)
    assert np.all(distance >= 0)
