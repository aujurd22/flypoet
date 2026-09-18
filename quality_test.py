"""Quality test: 3 arms x 5 prompts (poetry continuation + modern Chinese),
temperature 0.8, identical seeds. Outputs a side-by-side comparison file."""
import json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\user\flypoet")
import train_v2 as T

PROMPTS = [
    "《静夜思》床前明月光，",
    "《春晓》春眠不觉晓，",
    "登高望远，",
    "鲁迅先生说：",
    "深夜的城市，",
]

@torch.no_grad()
def gen(model, stoi, itos, prompt, n=120, temperature=0.8):
    ids = [stoi.get(c, 1) for c in prompt]
    idx = torch.tensor([ids], device=T.DEV)
    for _ in range(n):
        logits, _ = model(idx[:, -256:])
        probs = F.softmax(logits[0, -1].float() / temperature, dim=-1).unsqueeze(0)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos.get(str(i), "") for i in idx[0].tolist())

def main():
    corpus = T.Corpus()
    vj = json.load(open(os.path.join(T.DATA, "vocab.json"), encoding="utf-8"))
    stoi, itos = vj["stoi"], vj["itos"]
    out = open(r"D:\user\flypoet\quality_compare.txt", "w", encoding="utf-8")
    for arm, model_file, kw in (("std", "std_model.pt", None),
                                 ("flynetS", "flynetS_model.pt", {"impl": "torch"}),
                                 ("flynetS_adaptive", "flynetS_adaptive_model.pt", {"impl": "cuda"})):
        model = T.GPT(corpus.V, kwta_opts=kw).to(T.DEV)
        sd = torch.load(os.path.join("logs_v2", model_file), map_location=T.DEV, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        out.write(f"\n{'='*60}\n### {arm}\n{'='*60}\n")
        print(f"[{arm}] loaded", flush=True)
        for prompt in PROMPTS:
            text = gen(model, stoi, itos, prompt)
            out.write(f"\n-- prompt: {prompt}\n{text}\n")
            print(f"  {prompt} -> {text[:50]}", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()
    print("quality_compare.txt written")

if __name__ == "__main__":
    main()
