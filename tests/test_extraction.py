from geood.extraction import get_candidate_layers


def test_get_candidate_layers_32():
    layers = get_candidate_layers(32)
    assert layers == [0, 8, 16, 24, 31]


def test_get_candidate_layers_40():
    layers = get_candidate_layers(40)
    assert layers == [0, 10, 20, 30, 39]


def test_get_candidate_layers_small():
    layers = get_candidate_layers(4)
    assert len(layers) <= 4
