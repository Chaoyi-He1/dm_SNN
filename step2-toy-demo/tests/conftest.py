import sys
import pytest
sys.dont_write_bytecode = True

from toy_demo.config import tiny_config
from toy_demo.data import synthetic_slice, select_demo


@pytest.fixture(scope="session")
def cfg():
    return tiny_config()


@pytest.fixture(scope="session")
def sl():
    return synthetic_slice()


@pytest.fixture(scope="session")
def demo(sl, cfg):
    return select_demo(sl, cfg)
