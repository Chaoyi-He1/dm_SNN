"""二值(0/1)残差流模型 BinaryResidualLM 的转换实验:命令与 experiments/ternary_residual.py 相同,走 A0 -> A1 -> B -> C。

在 step2-toy-demo 目录执行(任意 cwd 也可,脚本自行切换);结果写到 --out/<name>/(默认 results/improve):
    .venv/bin/python experiments/binary_residual.py chain --name binres_sr_T2 --merge sr --levels 2     # iscan -> a1 -> bscan -> c
    .venv/bin/python experiments/binary_residual.py a1 --name binres_sr_T2_a1 --merge sr --levels 2 --fq 0.75
    .venv/bin/python experiments/binary_residual.py c --name binres_sr_T2_c --init results/improve/binres_sr_T2_a1/model.pt --lif-scale 1.5
    .venv/bin/python experiments/binary_residual.py iscan --merge xor --levels 1 --save results/improve/binres_xor_T1_iscan.json
    .venv/bin/python experiments/binary_residual.py bscan --init results/improve/binres_sr_T2_a1/model.pt --save results/improve/binres_sr_T2_bscan.json
加宽:--a0 results/improve/W_a0_d256/model.pt --d 256 --dk 64 --dv 64 --d-ff 512 --heads 4

各阶段的含义见 experiments/ternary_residual.py 的说明。合并方式 --merge:sr(置位 / 复位,子块输出仍三值,代理梯度)、src(同 sr,精确梯度)或 xor(翻转,子块输出无符号),
定义在 snn_spec/binary_residual.py。result.json 的 args 里 residual="binary",beam_eval 据此装载(kind = binres)。
另有一个对照用的 --merge satc:三值残差流(TernaryClampResidualLM,前向同三值残差的 sat,合并用精确梯度),
result.json 里 residual="ternary"、merge="satc",beam_eval 装载成 kind = ternres。

实现:五个命令函数直接用 experiments.ternary_residual 里的。它们通过那个模块的 make_model、calibrate_thresholds、save_stage
三个名字构造模型、标定阈值和存结果;binary_driver() 在运行期间把这三个名字换成本文件的版本,退出时还原,原文件不改动。
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
sys.path.insert(0, str(STEP2))

import experiments.ternary_residual as tr                                    # noqa: E402
from experiments.beam_eval import SIZE_KEYS                                   # noqa: E402
from experiments.convert_lab import _save                                     # noqa: E402
from snn_spec.binary_residual import MERGES                                   # noqa: E402
from toy_demo.binary_residual import BinaryResidualLM, TernaryClampResidualLM  # noqa: E402
from toy_demo.binary_residual import calibrate_thresholds as binary_calibrate  # noqa: E402
from toy_demo.ternary_residual import calibrate_thresholds as ternary_calibrate  # noqa: E402

TERNARY_CONTROL = "satc"                      # 三值残差流 + 精确梯度的合并,对照用


def make_model(cfg, sl, dev, merge, levels, leak):
    cls = TernaryClampResidualLM if merge == TERNARY_CONTROL else BinaryResidualLM
    return cls(cfg, sl.vocab_size, leak=leak, merge=merge, levels=levels).to(dev)


def calibrate_thresholds(model, tokens, quantile=0.75):
    """二值残差模型用带符号分位数的标定;三值对照沿用三值残差的标定。"""
    fn = binary_calibrate if isinstance(model, BinaryResidualLM) else ternary_calibrate
    return fn(model, tokens, quantile)


def save_stage(out_dir, m, res, args, cfg, extra):
    """补全 beam_eval.describe_checkpoint 需要的键(residual、尺寸取 cfg 的值)后存模型与 result.json。"""
    args.residual = "binary" if isinstance(m, BinaryResidualLM) else "ternary"
    for k, key in SIZE_KEYS:
        setattr(args, k, getattr(cfg, key))
    _save(out_dir, m, res, args, extra)


_HOOKS = dict(make_model=make_model, calibrate_thresholds=calibrate_thresholds, save_stage=save_stage)


@contextlib.contextmanager
def binary_driver():
    """在 with 块内让 experiments.ternary_residual 的命令函数构造 BinaryResidualLM;退出时还原。"""
    saved = {name: getattr(tr, name) for name in _HOOKS}
    for name, fn in _HOOKS.items():
        setattr(tr, name, fn)
    try:
        yield tr
    finally:
        for name, fn in saved.items():
            setattr(tr, name, fn)


def build_a1(a0, cfg, sl, tokens, dev, fq, merge, levels, seed=0, leak=0.0):
    """A0 -> A1:权重转移 + 阈值标定。返回 (模型, 阈值字典);merge = satc 时模型是三值对照 TernaryClampResidualLM。"""
    with binary_driver():
        return tr.build_a1(a0, cfg, sl, tokens, dev, fq, merge, levels, seed=seed, leak=leak)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", default="run")
    common.add_argument("--out", default="results/improve")
    common.add_argument("--a0", default="results/ckpt/a0.pt")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--eval-every", type=int, default=100)
    common.add_argument("--no-stop", action="store_true", help="忽略停止条件,跑满步数")
    common.add_argument("--lr", type=float, default=3e-3)
    common.add_argument("--steps", type=int, default=12000)
    for a, _ in SIZE_KEYS:
        common.add_argument("--" + a.replace("_", "-"), dest=a, type=int, default=None,
                            help="模型尺寸(默认 ToyConfig;c/bscan 从 --init 继承)")

    def model_args(p):
        p.add_argument("--merge", default="sr", choices=list(MERGES) + [TERNARY_CONTROL], help="残差合并方式(satc 是三值对照)")
        p.add_argument("--levels", type=int, default=1, help="每 token 最多几个脉冲")

    p1 = sub.add_parser("a1", parents=[common])
    model_args(p1)
    p1.add_argument("--fq", type=float, default=0.75, help="阈值标定分位数")
    p1.set_defaults(fn=tr.cmd_a1)
    pc = sub.add_parser("c", parents=[common])
    pc.add_argument("--init", required=True, help="A1 的 model.pt(merge、levels、fq、尺寸从同目录 result.json 继承)")
    pc.add_argument("--lif-scale", type=float, default=1.0, help="训练前只把 LIF 阈值乘以该系数")
    pc.set_defaults(fn=tr.cmd_c)
    pi = sub.add_parser("iscan", parents=[common])
    model_args(pi)
    pi.add_argument("--fq-list", default="0.5,0.75,0.9")
    pi.add_argument("--save", default=None)
    pi.set_defaults(fn=tr.cmd_iscan)
    pb = sub.add_parser("bscan", parents=[common])
    pb.add_argument("--init", required=True)
    pb.add_argument("--scale-list", default="1,1.5,2,2.25,2.5,3,4")
    pb.add_argument("--save", default=None)
    pb.set_defaults(fn=tr.cmd_bscan)
    pn = sub.add_parser("chain", parents=[common])
    model_args(pn)
    pn.add_argument("--fq-list", default="0.5,0.75,0.9")
    pn.add_argument("--scale-list", default="1,1.5,2,2.25,2.5,3,4")
    pn.set_defaults(fn=tr.cmd_chain)
    args = ap.parse_args(argv)
    with binary_driver():
        args.fn(args)


if __name__ == "__main__":
    main()
