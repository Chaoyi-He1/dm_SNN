"""Run pinned upstream code independently, not a handwritten translation."""
import pytest
from experiments.check_upstream import case, chunk_case


@pytest.mark.parametrize('length', [1, 7, 65])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('mode', ['normalized', 'continuous', 'spiking'])
def test_core_matches_pinned_qwen35(length, initial, mode):
    case(length, initial, mode)


def test_upstream_chunk_boundary_matches_recurrent():
    chunk_case()
