import torch
import torch.nn as nn
from bielik_hallu.extract.model import ActivationCapturer


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.act_fn(x)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()
    def forward(self, x):
        return self.mlp(x)


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class _Model(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.model = _Inner(n)
    def forward(self, x):
        return self.model(x)


def test_capturer_records_per_layer():
    m = _Model(n=3)
    x = torch.randn(1, 4, 8)  # (batch, seq, hidden)
    with ActivationCapturer(m) as cap:
        m(x)
        assert set(cap.activations.keys()) == {0, 1, 2}
        assert cap.activations[0].shape[-1] == 8


def test_hooks_removed_after_exit():
    m = _Model(n=2)
    with ActivationCapturer(m) as cap:
        pass
    assert len(cap._handles) == 0
