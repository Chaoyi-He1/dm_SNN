"""每 token 能耗的纯公式:memristor 电路(按部件)、稠密 int8 数字 ASIC、事件驱动的数字脉冲网络。
输入是模型尺寸 cfg、实测事件计数 events 与量化前的权重矩阵 mats,输出是 SI 单位(焦耳)的分解字典;
所有参数都从 energy_params 取(每项带来源),本模块只有 numpy,不读写文件。"""
import toy_demo  # noqa: F401  先导入它才能让 step1 的 experiments.spice 可见
import numpy as np

from experiments.energy_params import v
from experiments.spice import crossbar as xb

_XBAR_KEYS = ("conduction", "gate", "wire", "leak_off", "integrator")


def mac_counts(cfg, vocab):
    """每 token 的乘加数(各块求和)与权重参数量。权重流量 params_stream 不含嵌入表(每 token 只查 d 个值)。"""
    d, nb, H, dk, dv, dff = cfg.d, cfg.n_blocks, cfg.n_heads, cfg.dk, cfg.dv, cfg.d_ff
    qkv = d * H * (2 * dk + dv)                      # Wq、Wk、Wv;toy 配置里 H·dk = H·dv = d,即 3·d·d
    o = H * dv * d
    ffn13 = 2 * d * dff
    ffn2 = dff * d
    gates = 2 * d * H
    core = H * 4 * dk * dv                           # 检索、写入、读出、衰减各 dk·dv
    out = dict(proj_qkv=nb * qkv, proj_o=nb * o, ffn_13=nb * ffn13, ffn_2=nb * ffn2, gates=nb * gates,
               core=nb * core, head=d * vocab)
    out["total_macs"] = sum(out.values())
    out["params_stream"] = nb * (qkv + o + ffn13 + ffn2 + gates + 2 * H) + d * vocab     # 门偏置 2·H;含输出头
    out["params_embedding"] = vocab * d
    return out


def row_conductance(W, levels, G_on, G_off):
    """每个输入行的总电导 Σ_j (G⁺ + G⁻)[j, i];W: [d_out, d_in],返回 [d_in]。
    量化沿用 crossbar.quantize_weights;电导台阶由这里的 G_on、G_off 重算(crossbar.conductance_targets 固定用测试台常数)。"""
    npos, nneg, _, _ = xb.quantize_weights(W, levels)
    dG = (G_on - G_off) / (levels - 1)
    return ((G_off + npos * dG) + (G_off + nneg * dG)).sum(0)


def crossbar_energy(c_rows, grow, d_out, col, n_pulses_window=1):
    """一个 1T1R 差分 crossbar 每 token 的能量。c_rows[i] = 行 i 每 token 的平均脉冲数,grow[i] = 行 i 的总电导,
    d_out = 列数(每个权重 2 个 cell)。"""
    c, g = np.asarray(c_rows, dtype=float), np.asarray(grow, dtype=float)
    V, t = v(col, "V_read"), v(col, "t_read")
    pulses = float(c.sum())                                              # 每 token 的驱动行次数
    cells = 2 * d_out
    out = dict(
        conduction=V * V * t * float((c * g).sum()),
        gate=pulses * cells * v(col, "C_gate") * v(col, "V_sel") ** 2,
        wire=pulses * cells * v(col, "C_wire") * V * V,
        leak_off=max(0.0, cells * len(c) - pulses * cells) * V * v(col, "I_off") * t,
        integrator=d_out * v(col, "I_int") * v(col, "V_dd") * t * n_pulses_window,
    )
    out["total"] = sum(out[k] for k in _XBAR_KEYS)
    return out


def _event_add_energy(n_adds, col, batch):
    """事件驱动数字加法的三项:加法本身、8 bit 权重取数(按 batch 摊销)、激活读写(每次加法 2 次 8 bit 近端访存)。"""
    return dict(add=n_adds * v(col, "E_add16"),
                weight=n_adds * 8 * v(col, "E_sram_far_bit") / batch,
                act=n_adds * 2 * 8 * v(col, "E_sram_near_bit"))


def memristor_energy(events, mats, cfg, vocab, col, head):
    """memristor 电路每 token 的能量,按部件与按矩阵分解。head: "crossbar_adc"(输出头放 crossbar,3226 列各一次 ADC)
    或 "digital_event"(输出头用事件驱动的数字加法,batch 1)。"""
    if head not in ("crossbar_adc", "digital_event"):
        raise ValueError(f"head 必须是 crossbar_adc 或 digital_event,收到 {head!r}")
    G_on, G_off, levels = v(col, "G_on"), v(col, "G_off"), int(v(col, "levels"))
    n_win = events["levels"]
    by_matrix = {}
    for name, W in mats.items():
        d_out = W.shape[0]
        if name == "head" and head == "digital_event":
            by_matrix[name] = dict.fromkeys(("total",) + _XBAR_KEYS, 0.0)          # 输出头不走 crossbar
            continue
        c = events["embed_rows"] if name == "embed" else events["rows"][name]
        if len(c) != W.shape[1]:
            raise ValueError(f"{name}: 事件行数 {len(c)} 与矩阵输入维 {W.shape[1]} 不一致")
        by_matrix[name] = crossbar_energy(c, row_conductance(W, levels, G_on, G_off), d_out, col, n_win)

    def total_of(key, names=None):
        return sum(e[key] for n, e in by_matrix.items() if names is None or n in names)

    blocks = [n for n in by_matrix if n.startswith("blocks.")]
    gain, g_cell = events["gain"], 0.5 * v(col, "C_s") * v(col, "V_s") ** 2
    cells = cfg.n_blocks * cfg.n_heads * cfg.dk * cfg.dv
    gain_cells = (cells * (1 - gain["alpha_mean"] ** 2) * g_cell * gain["s_rms_norm"] ** 2      # 泄漏
                  + 2 * gain["k_rows"] * cfg.dv * g_cell                                        # 检索 + 写入
                  + gain["q_rows"] * cfg.dv * g_cell)                                           # 读出

    head_digital = 0.0
    if head == "digital_event":
        e = _event_add_energy(float(np.sum(events["rows"]["head"])) * vocab, col, batch=1)
        head_digital = e["add"] + e["weight"] + e["act"]

    comp = dict(
        crossbar_blocks=total_of("conduction", blocks),
        crossbar_head=total_of("conduction", {"head"}),
        embed=total_of("conduction", {"embed"}),
        gate_wire=total_of("gate") + total_of("wire"),
        integrator=total_of("integrator"),
        leak_off=total_of("leak_off"),
        adm_spikes=events["spikes_lif"] * v(col, "E_spike"),
        adm_leak=events["n_lif"] * v(col, "V_gs") ** 2 / v(col, "R_c") * v(col, "t_token"),
        comparators=events["cmp_decisions"] * v(col, "E_cmp"),
        gain_cells=gain_cells,
        adc=vocab * v(col, "E_adc") if head == "crossbar_adc" else 0.0,
        head_digital=head_digital,
        control=v(col, "E_ctrl"),
    )
    comp = {k: float(x) for k, x in comp.items()}
    return dict(by_component=comp, by_matrix=by_matrix, total=sum(comp.values()))


def _core_energy(cfg, col):
    """GDN 状态递推(检索、写入、读出、衰减)的乘加与其激活访存;稠密与事件驱动数字侧相同(门值与状态是连续量)。"""
    macs = cfg.n_blocks * cfg.n_heads * 4 * cfg.dk * cfg.dv
    return macs * (v(col, "E_mul8") + v(col, "E_add32")) + macs * 2 * 8 * v(col, "E_sram_near_bit")


def _state_and_embed(cfg, col):
    """状态 S 每 token 一次读一次写(8 bit,近端 SRAM)与嵌入查表(d × 8 bit,片上 SRAM);稠密与事件驱动相同。"""
    state = cfg.n_blocks * cfg.n_heads * cfg.dk * cfg.dv * 2 * 8 * v(col, "E_sram_near_bit")
    embed = cfg.d * 8 * v(col, "E_sram_far_bit")
    return state, embed


def digital_dense_energy(cfg, vocab, col, batch):
    """稠密 int8 数字 ASIC 每 token 的能量:乘加、激活访存、权重取数(batch 摊销)、嵌入查表、GDN 状态访存。"""
    m = mac_counts(cfg, vocab)
    state, embed = _state_and_embed(cfg, col)
    out = dict(mac=m["total_macs"] * (v(col, "E_mul8") + v(col, "E_add32")),
               act=m["total_macs"] * 2 * 8 * v(col, "E_sram_near_bit"),
               weight=m["params_stream"] * 8 * v(col, "E_sram_far_bit") / batch,
               embed=embed, state=state)
    out["total"] = sum(out.values())
    return out


def digital_event_energy(events, cfg, vocab, col, batch):
    """事件驱动数字脉冲网络每 token 的能量:只对被驱动的行做 16 bit 加法、只取这些行的 8 bit 权重;
    LIF 神经元每次更新一次泄漏乘法加一次比较(n_lif 个);GDN 递推 core、状态与嵌入同稠密。"""
    H = cfg.n_heads
    d_out = {"qkv": H * (2 * cfg.dk + cfg.dv), "gates": 2 * H, "o": cfg.d, "ffn13": 2 * cfg.d_ff, "w2": cfg.d}
    n_adds = 0.0
    for name, c in events["rows"].items():
        n_adds += float(np.sum(c)) * (vocab if name == "head" else d_out[name.split(".")[-1]])
    out = _event_add_energy(n_adds, col, batch)
    state, embed = _state_and_embed(cfg, col)
    out.update(neurons=events["n_lif"] * (v(col, "E_mul8") + v(col, "E_add16")),
               core=_core_energy(cfg, col), state=state, embed=embed)
    out["total"] = sum(out.values())
    return out
