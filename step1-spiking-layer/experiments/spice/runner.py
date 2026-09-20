"""ngspice 批处理运行与结果解析。"""
import re
import subprocess
import tempfile
from pathlib import Path


def run_ngspice(netlist: str, files=(), timeout=600, links=()):
    """在临时目录里以批处理模式运行 ngspice。返回 (日志文本, {文件名: 内容})。
    links:要以相对路径出现在临时目录里的外部文件(如 PDK 模型库),用符号链接放入。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        for src in links:
            (p / Path(src).name).symlink_to(Path(src).resolve())
        (p / "cir.cir").write_text(netlist)
        r = subprocess.run(["ngspice", "-b", "-o", "log.txt", "cir.cir"], cwd=d,
                           capture_output=True, text=True, timeout=timeout)
        log = ""
        if (p / "log.txt").exists():
            log += (p / "log.txt").read_text(errors="ignore")
        log += "\n" + r.stdout + "\n" + r.stderr
        out = {f: (p / f).read_text() for f in files if (p / f).exists()}
    return log, out


def parse_meas(log: str) -> dict:
    """解析 `meas` 的输出行,如 `s_end = 1.250000e+00 at= 1.2e-06`。"""
    res = {}
    for name, val in re.findall(r"^\s*(\w+)\s*=\s*([-+0-9.]+(?:[eE][-+]?\d+)?)", log, re.M):
        try:
            res[name] = float(val)
        except ValueError:
            pass
    return res


def parse_wrdata(text: str):
    """解析 `set wr_singlescale` + `set wr_vecnames` 下 wrdata 的输出:首行为列名,其余为数值。"""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    names = lines[0].split()
    rows = [[float(x) for x in l.split()] for l in lines[1:]]
    cols = list(zip(*rows))
    return {n: list(c) for n, c in zip(names, cols)}
