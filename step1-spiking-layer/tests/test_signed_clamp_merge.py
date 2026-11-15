"""三值对照用的合并 satc:前向与 ResidualMerge("sat") 相同,反向是 clamp 的精确梯度。"""
import itertools
import torch

from snn_spec.binary_residual import SignedClampMerge, clamp_exact
from snn_spec.ternary_residual import ResidualMerge


def test_satc_forward_equals_sat_on_every_level_grid():
    for T in (1, 2, 4):
        vals = tuple(j / T for j in range(-T, T + 1))
        g = torch.tensor(list(itertools.product(vals, repeat=2)))
        r, y = g[:, 0], g[:, 1]
        out = SignedClampMerge(levels=T)(r, y)
        assert torch.equal(out, torch.clamp(r + y, -1.0, 1.0)), T
        assert torch.equal(out, ResidualMerge("sat", levels=T)(r, y)), T


def test_satc_gradient_is_one_unless_saturated():
    cases = [((0.0, 0.0), 1.0), ((1.0, 0.0), 1.0), ((-1.0, 0.0), 1.0), ((1.0, -1.0), 1.0), ((0.0, 1.0), 1.0),
             ((1.0, 1.0), 0.0), ((-1.0, -1.0), 0.0), ((1.0, 0.5), 0.0), ((0.5, 0.5), 1.0)]
    for (rv, yv), want in cases:
        r, y = torch.tensor([rv], requires_grad=True), torch.tensor([yv], requires_grad=True)
        SignedClampMerge(levels=2)(r, y).sum().backward()
        assert (r.grad.item(), y.grad.item()) == (want, want), (rv, yv)


def test_sat_surrogate_is_weaker_than_the_exact_gradient_on_the_identity_path():
    """原来的 sat(代理梯度)在 r = y = 0 处的梯度是 2 / (1 + π²) ≈ 0.18,satc 是 1:这就是要控制的差别。"""
    r, y = torch.zeros(1, requires_grad=True), torch.zeros(1)
    ResidualMerge("sat")(r, y).sum().backward()
    assert abs(r.grad.item() - 2.0 / (1.0 + torch.pi ** 2)) < 1e-6
    r2 = torch.zeros(1, requires_grad=True)
    SignedClampMerge()(r2, y).sum().backward()
    assert r2.grad.item() == 1.0


def test_interface_matches_the_other_merges_and_clamp_exact_adds_nothing_to_the_value():
    m = SignedClampMerge(levels=2)
    assert m.merge == "satc" and m.levels == 2 and m.thr.item() == 1.0
    assert not m.log_thr.requires_grad and list(m.state_dict()) == ["log_thr"]
    u = torch.linspace(-3, 3, 25, requires_grad=True)
    out = clamp_exact(u, -1.0, 1.0)
    assert torch.equal(out.detach(), torch.clamp(u.detach(), -1.0, 1.0))
    out.sum().backward()
    assert torch.equal(u.grad, ((u.detach() >= -1) & (u.detach() <= 1)).float())
