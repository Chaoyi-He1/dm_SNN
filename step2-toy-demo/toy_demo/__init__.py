"""玩具模型 demo:从零训练的纯 GDN 小模型,浮点 → 可脉冲化 → 折叠 T=1 → 行为级电路。设计见 ../../toy-demo-plan.md。

snn_spec 总是取同一个代码树里的 ../step1-spiking-layer:venv 里以可编辑方式装的 snn_spec 只指向某一个 checkout,
从别的 checkout(例如 git worktree)运行时会导入那边的旧代码。这里在第一次导入 snn_spec 之前把同级目录放到 sys.path
最前面(setuptools 的可编辑查找器排在普通路径查找之后,所以同级目录优先),不需要再设 PYTHONPATH。
"""
import sys
import warnings
from pathlib import Path

STEP1 = Path(__file__).resolve().parents[2] / "step1-spiking-layer"
if (STEP1 / "snn_spec").is_dir():
    if str(STEP1) not in sys.path:
        sys.path.insert(0, str(STEP1))
    _loaded = sys.modules.get("snn_spec")
    if _loaded is not None and STEP1 not in Path(_loaded.__file__).resolve().parents:
        warnings.warn(f"snn_spec 已从 {_loaded.__file__} 导入,不是同一代码树的 {STEP1};请先导入 toy_demo 再导入 snn_spec",
                      stacklevel=2)
