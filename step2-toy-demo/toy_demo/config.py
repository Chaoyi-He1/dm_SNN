"""玩具模型 demo 的全部配置:尺寸、超参数、非理想性数值、种子。数值来源见 ../../toy-demo-plan.md 第 4–6 节。"""
from dataclasses import dataclass, asdict


@dataclass
class ToyConfig:
    # 数据(第 3 节)
    slice_dir: str = "data/tinystories-slice"
    target_sentences: int = 4000
    min_len: int = 2                 # 不含特殊 token
    max_len: int = 40
    n_demo: int = 5
    demo_prefix_len: int = 5
    demo_len_min: int = 12
    demo_len_max: int = 16
    demo_seed: int = 0
    # 模型(第 4 节)
    d: int = 128
    n_blocks: int = 2
    n_heads: int = 4
    dk: int = 32
    dv: int = 32
    d_ff: int = 256
    alpha_min: float = 0.5
    alpha_max: float = 0.99
    leak: float = 0.9
    fire_quantile: float = 0.75
    calib_sentences: int = 256
    # 训练(第 5 节)
    batch_size: int = 64
    clip: float = 1.0
    lr_a0: float = 3e-3
    steps_a0: int = 6000
    lr_a1: float = 3e-3
    steps_a1: int = 2000
    lr_c: float = 1e-3
    steps_c: int = 1500
    eval_every: int = 100
    plateau_steps: int = 500
    plateau_delta: float = 0.001     # 0.1 个百分点
    seeds: tuple = (0, 1, 2)
    max_new_tokens: int = 40
    # 电路(第 6 节)
    n_levels: int = 16
    sigma_g: float = 0.02
    read_noise: float = 0.01
    write_err: float = 0.02
    read_err: float = 0.02
    eps_hold: float = 0.999
    cmp_offset_frac: float = 0.02
    sigma_g_sweep: tuple = (0.0, 0.02, 0.05, 0.10)
    n_compare_sentences: int = 200
    compare_seed: int = 0
    # 输出
    results_dir: str = "results"

    def as_dict(self):
        return asdict(self)


def tiny_config(**overrides):
    """测试用的微型配置:秒级训练、合成切片。"""
    base = dict(d=16, n_blocks=2, n_heads=2, dk=4, dv=4, d_ff=32, batch_size=8,
                steps_a0=20, steps_a1=10, steps_c=10, eval_every=5, plateau_steps=10,
                calib_sentences=32, n_compare_sentences=8, max_new_tokens=8, results_dir="results/tiny")
    base.update(overrides)
    return ToyConfig(**base)
