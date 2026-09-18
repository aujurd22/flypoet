"""Strict audit tests for the FlyPoet training stack.

Run:  python -m pytest tests/ -q
These tests encode the failure modes we actually care about as a training
researcher: data leakage, silent dtype/cast breakage, gradient-flow holes,
metric validity, and fairness between arms.
"""
import json, os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, r"D:\user\flypoet")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- corpus tests

@pytest.mark.skipif(not os.path.exists(r"D:\user\flypoet\data_v2\train.bin"),
                    reason="corpus not prepared yet")
class TestCorpus:
    def test_token_range(self):
        vj = json.load(open(r"D:\user\flypoet\data_v2\vocab.json", encoding="utf-8"))
        V = len(vj["stoi"])
        tr = np.fromfile(r"D:\user\flypoet\data_v2\train.bin", dtype=np.uint16)
        va = np.fromfile(r"D:\user\flypoet\data_v2\val.bin", dtype=np.uint16)
        assert tr.max() < V and va.max() < V

    def test_val_token_count(self):
        va = np.fromfile(r"D:\user\flypoet\data_v2\val.bin", dtype=np.uint16)
        assert len(va) > 50_000, "val too small for stable 24-batch eval"

    def test_vocab_roundtrip(self):
        vj = json.load(open(r"D:\user\flypoet\data_v2\vocab.json", encoding="utf-8"))
        stoi, itos = vj["stoi"], vj["itos"]
        for c, i in list(stoi.items())[:100]:
            assert itos[str(i)] == c

    def test_val_unk_rate_not_pathological(self):
        vj = json.load(open(r"D:\user\flypoet\data_v2\vocab.json", encoding="utf-8"))
        va = np.fromfile(r"D:\user\flypoet\data_v2\val.bin", dtype=np.uint16)
        unk_rate = float((va == 1).mean())
        # val built from chars of train vocab -> unk should be rare
        assert unk_rate < 0.05, f"val unk rate {unk_rate:.1%} suggests vocab leakage miss"


# ---------------------------------------------------------------- KWTA tests

class TestKWTA:
    def _layer(self, impl):
        from train_v2 import KWTA
        return KWTA(0.10, impl=impl)

    @pytest.mark.parametrize("impl", ["torch", "cuda"])
    def test_sparsity_exactly_ten_pct(self, impl):
        if impl == "cuda" and not os.environ.get("VCVARS_OK"):
            pytest.skip("cuda ext needs MSVC env (run pytest via build_kwta.bat)")
        layer = self._layer(impl).to(DEV)
        x = torch.randn(4, 32, 768, device=DEV)
        y = layer(x)
        sparsity = float((y == 0).float().mean())
        assert 0.85 <= sparsity <= 0.95, f"sparsity {sparsity:.2f} off from 90% target"

    @pytest.mark.parametrize("impl", ["torch", "cuda"])
    def test_gradient_flows_only_through_kept(self, impl):
        if impl == "cuda" and not os.environ.get("VCVARS_OK"):
            pytest.skip("cuda ext needs MSVC env")
        layer = self._layer(impl).to(DEV)
        x = torch.randn(2, 8, 768, device=DEV, requires_grad=True)
        y = layer(x)
        g = torch.randn_like(y)
        y.backward(g)
        kept = (y != 0)
        dropped_grads = x.grad[~kept]
        assert dropped_grads.abs().sum() == 0, "dropped channels must get zero grad"

    def test_adaptive_vs_exact_overlap_reasonable(self):
        """CUDA 编译需要 MSVC 环境（where cl）；无环境时跳过。"""
        import shutil
        if shutil.which("cl") is None:
            pytest.skip("MSVC cl not on PATH (run via build_kwta.bat env)")
        from adaptive_kwta import adaptive_kwta
        x = torch.randn(8, 64, 768, device=DEV)
        y_ad = adaptive_kwta(x, 0.10)
        d = x.shape[-1]; k = int(d * 0.10)
        thr = torch.kthvalue(x, d - k + 1, dim=-1, keepdim=True).values
        y_ex = x * (x >= thr)
        inter = ((y_ad != 0) & (y_ex != 0)).sum().item()
        n_ad = (y_ad != 0).sum().item()
        assert n_ad > 0
        assert inter / n_ad > 0.70, "adaptive retained set deviates too much from top-k"

    def test_kwta_edge_all_zero(self):
        layer = self._layer("torch").to(DEV)
        x = torch.zeros(1, 4, 768, device=DEV)
        y = layer(x)          # ties everywhere: kthvalue threshold = 0, x>=0 keeps all
        assert torch.isfinite(y).all()

    def test_kwta_deterministic(self):
        layer = self._layer("torch").to(DEV)
        x = torch.randn(2, 4, 768, device=DEV)
        torch.manual_seed(0)
        y1 = layer(x)
        y2 = layer(x)
        assert torch.equal(y1, y2)


# ---------------------------------------------------------------- model tests

class TestModel:
    def _model(self, kwta_opts=None, layers=2, d=64):
        from train_v2 import GPT
        return GPT(V=300, d=d, layers=layers, heads=4, ffn_h=96, seq=64,
                   kwta_opts=kwta_opts).to(DEV)

    def test_init_distribution(self):
        torch.manual_seed(0)
        m = self._model()
        # only the weights we explicitly init (MHA in_proj keeps xavier default)
        lins = [p for n, p in m.named_parameters()
                if p.dim() == 2 and n.endswith("ffn.0.weight")]
        stds = [p.std().item() for p in lins]
        assert all(0.005 < s < 0.05 for s in stds), f"init stds off: {stds[:3]}"

    def test_residual_proj_scaled_down(self):
        torch.manual_seed(0)
        m = self._model(layers=4)
        res = [p for n, p in m.named_parameters() if "ffn.2.weight" in n]
        import math
        want = 0.02 / math.sqrt(2 * 4)
        for p in res:
            assert abs(p.std().item() - want) < want * 0.5

    def test_weight_tying(self):
        m = self._model()
        assert m.head.weight is m.emb.weight

    def test_fairness_same_param_count(self):
        import io, contextlib
        bufs = {}
        for kw in ({"impl": "torch"}, {"impl": "cuda"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m = self._model(kwta_opts=kw)
            bufs[str(kw)] = sum(p.numel() for p in m.parameters())
        vals = list(bufs.values())
        assert vals[0] == vals[1], "arms must have identical parameter counts"

    def test_loss_decreases_on_overfit(self):
        """100 steps on one fixed batch: loss must drop sharply. This is the
        single best smoke test for 'training machinery actually works'."""
        for kw in (None, {"impl": "torch"}):
            torch.manual_seed(0)
            m = self._model(kwta_opts=kw, layers=2, d=64)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            x = torch.randint(0, 300, (4, 64), device=DEV)
            y = x.clone()
            first = None
            for _ in range(100):
                opt.zero_grad()
                _, loss = m(x, y)
                loss.backward()
                opt.step()
                first = first or loss.item()
            assert loss.item() < first * 0.3, \
                f"loss {first:.2f} -> {loss.item():.2f}: training machinery broken"
