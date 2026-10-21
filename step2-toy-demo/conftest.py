import sys
sys.dont_write_bytecode = True  # 与 step1 一致,不写 __pycache__

import toy_demo  # noqa: E402,F401  先把同一代码树的 ../step1-spiking-layer 放到 sys.path 最前面(见 toy_demo/__init__.py)
