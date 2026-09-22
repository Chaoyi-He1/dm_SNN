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


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, cfg, sl):
    from toy_demo.run import run_pipeline
    out = tmp_path_factory.mktemp("tiny")
    results = run_pipeline(cfg, sl, out, log=lambda *_: None)
    return cfg, sl, out, results
