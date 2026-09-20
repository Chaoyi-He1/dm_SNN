"""S0.4 的稳定性命题与压力测试 1–3:直接向递推核心注入 k, v, q 和门值,绕过编码器。"""
import math
import torch

from .gdn import SpikingGDN


def state_bound(dv, alpha_max):
    """命题:S_0 = 0、||k||^2 <= 1、beta in (0,1)、alpha <= alpha_max < 1、v 三值 ⇒ sup ||S||_F <= sqrt(dv)/(1-alpha_max)。"""
    return math.sqrt(dv) / (1.0 - alpha_max)


@torch.no_grad()
def core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, seed=0, device="cpu"):
    """单头。返回三个布尔结论和 max_norm_over_bound(测试 2、3 中 ||S||_F / 界 的最大值)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, H = 1, 1
    bound = state_bound(dv, alpha)
    ck = 1.0 / math.sqrt(dk)
    a = torch.full((B, H), alpha, device=device)
    b = torch.full((B, H), beta, device=device)
    q = torch.ones(B, H, dk, device=device)

    def rnd_v():
        return torch.randint(-1, 2, (B, H, dv), generator=g).float().to(device)

    res = {}
    # 1) 全零 key 200 步:||S_t|| = alpha^t ||S_0||,相对偏差 <= 1%
    # Use float64 for decay: 0.5**200 is below float32 range.
    S = torch.randn(B, H, dk, dv, generator=g, dtype=torch.float64).to(device); S0 = S.norm().item(); ok = True
    for t in range(1, 201):
        _, S = SpikingGDN.core(torch.zeros_like(q, dtype=torch.float64), rnd_v().double(),
                               q.double(), a.double(), b.double(), S)
        expect = (alpha ** t) * S0
        ok &= abs(S.norm().item() - expect) <= 0.01 * expect
    res["zero_key_follows_alpha_product"] = ok
    # 2) 全一 key(||k||^2 = 1)200 步:不超过命题的界
    S = torch.zeros(B, H, dk, dv, device=device); ok = True; ratio = 0.0
    k = torch.ones(B, H, dk, device=device) * ck
    for _ in range(200):
        _, S = SpikingGDN.core(k, rnd_v(), q, a, b, S)
        ratio = max(ratio, S.norm().item() / bound)
        ok &= S.norm().item() <= bound * 1.05
    res["all_one_key_bounded"] = ok
    # 3) 连续稠密随机 key(n = dk)1000 步:有界
    S = torch.zeros(B, H, dk, dv, device=device); ok = True
    for _ in range(1000):
        s = torch.randint(0, 2, (B, H, dk), generator=g).float().to(device) * 2 - 1
        _, S = SpikingGDN.core(s * ck, rnd_v(), q, a, b, S)
        ratio = max(ratio, S.norm().item() / bound)
        ok &= S.norm().item() <= bound * 1.05
    res["dense_key_bounded"] = ok
    res["max_norm_over_bound"] = ratio
    res["bound"] = bound
    return res


def main():
    for alpha, beta in [(0.99, 0.5), (0.9, 0.9), (0.5, 0.99)]:
        r = core_stress_tests(alpha=alpha, beta=beta)
        print(f"alpha={alpha:<5} beta={beta:<5} bound={r['bound']:.2f} "
              f"max||S||/bound={r['max_norm_over_bound']:.3f} "
              f"test1={r['zero_key_follows_alpha_product']} test2={r['all_one_key_bounded']} "
              f"test3={r['dense_key_bounded']}")


if __name__ == "__main__":
    main()
