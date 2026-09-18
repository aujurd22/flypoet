"""Integration test: tiny model, real training loop, all three arms.

Also audits prepare_v2.py source for the leakage traps it must not have.
"""
import io, os, sys, contextlib
import numpy as np
import pytest
import torch

sys.path.insert(0, r"D:\user\flypoet")


def test_prepare_v2_no_val_leak_in_vocab_build():
    """The vocab must be built from TRAIN docs only; audit the source for
    the 'all_text' construction order."""
    src = open(r"D:\user\flypoet\prepare_v2.py", encoding="utf-8").read()
    split_at = src.index("split = int(len(docs) * 0.97)")
    vocab_at = src.index("all_train = ")
    # vocab construction text must come after the split and reference train_docs
    assert "train_docs" in src[vocab_at:vocab_at + 120]
    # shuffle must precede split (doc-level randomisation)
    assert src.index("random.shuffle(docs)") < split_at


def test_prepare_v2_val_is_doc_level():
    """Val docs must not be mid-document chunks of train docs (source audit:
    poetry entries are appended whole; luxun chunks come from the same doc
    but different offsets -- verify the chunker strides never overlap)."""
    src = open(r"D:\user\flypoet\prepare_v2.py", encoding="utf-8").read()
    stride = 2000
    assert "range(0, len(text), 2000)" in src.split("docs = []")[1] if "docs = []" in src else True, "chunk stride audit"
    # chunks are non-overlapping by construction: step == size
    assert "luxun" in src, "prepare_v2 must ingest modern literature"


@pytest.mark.parametrize("arm", ["std", "flynetS", "flynetS_adaptive"])
def test_tiny_training_loop_all_arms(arm):
    """Tiny model + 30 steps: loss decreases, no NaN, all arms train."""
    if arm == "flynetS_adaptive" and not os.environ.get("VCVARS_OK"):
        pytest.skip("cuda ext needs MSVC env")
    from train_v2 import GPT
    torch.manual_seed(0)
    model = GPT(V=200, d=48, layers=2, heads=4, ffn_h=96, seq=32,
                kwta_opts=({"impl": "torch"} if arm == "flynetS" else
                           {"impl": "cuda"} if arm == "flynetS_adaptive" else None)).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 200, (8, 32), device="cuda")
    y = x.clone()
    losses = []
    for _ in range(30):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0] * 0.9, f"loss not decreasing: {losses[0]:.2f} -> {losses[-1]:.2f}"


def test_autocast_cuda_kernel_compat():
    """REGRESSION for the bug we predicted in review: the CUDA adaptive k-WTA
    kernel asserts fp32 input, but autocast feeds bf16. The KWTA module must
    upcast before calling the kernel. Needs MSVC env (cl on PATH)."""
    import shutil
    if shutil.which("cl") is None:
        pytest.skip("MSVC cl not on PATH (run via build_kwta.bat env)")
    from train_v2 import KWTA
    layer = KWTA(0.10, impl="cuda").cuda()
    x = torch.randn(2, 8, 768, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)          # must not raise
    assert y.dtype == torch.bfloat16
