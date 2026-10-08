
# -*- coding: utf-8 -*-
"""Build the seven class prototypes for the LAION ViT-B/32 text encoder.

Same prompt set as the original features/text_prototypes.npz (four prompts per class), so the
CLIP-Adapter anchor of the third-encoder axis is the text side of that encoder, not of another one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import open_clip

WEIGHTS = "D:/ResearchVault/99system/models/open_clip/ViT-B-32-laion2B-fixed.pt"
OUT = Path("D:/ResearchVault/99system/data/rb-tta-fer-fg2027/features_vitb32laion/text_prototypes.npz")
PROMPTS = {
    "angry": ["a photo of an angry facial expression", "a face showing anger",
              "furrowed brows and tightened lips", "an irritated face with a tense mouth"],
    "disgust": ["a photo of a disgusted facial expression", "a face showing disgust",
                "a wrinkled nose and raised upper lip", "a repulsed face with a grimace"],
    "fear": ["a photo of a fearful facial expression", "a face showing fear",
             "wide eyes and raised eyebrows with an open mouth", "a terrified face with tense features"],
    "happy": ["a photo of a happy facial expression", "a smiling face",
              "raised cheeks and a smiling mouth", "a joyful face with upturned lips"],
    "sad": ["a photo of a sad facial expression", "a face showing sadness",
            "downturned mouth and drooping eyelids", "a sorrowful face with lowered brows"],
    "surprise": ["a photo of a surprised facial expression", "a face showing surprise",
                 "wide eyes, raised eyebrows, and an open mouth", "an astonished face with arched brows"],
    "neutral": ["a photo of a neutral facial expression", "a face with no strong expression",
                "relaxed facial muscles and a neutral mouth", "an emotionless face with a calm expression"],
}
ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=WEIGHTS, device="cuda",
                                                        weights_only=False)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    protos, names = [], []
    with torch.inference_mode():
        for c in ORDER:
            t = tok(PROMPTS[c]).cuda()
            f = model.encode_text(t)
            f = f / f.norm(dim=-1, keepdim=True)
            p = f.mean(0)
            p = p / p.norm()
            protos.append(p.cpu().numpy().astype(np.float32))
            names.append(c)
    P = np.stack(protos)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, class_names=np.asarray(names), prototypes=P,
                        prompts_json=np.asarray(json.dumps(PROMPTS)))
    print("written", OUT, P.shape, "norms", np.linalg.norm(P, axis=1))


if __name__ == "__main__":
    main()

