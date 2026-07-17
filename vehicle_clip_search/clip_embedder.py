"""Local CLIP embeddings (open_clip, ViT-B-32). No API key needed."""

import torch
import open_clip
from PIL import Image

from .config import CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load():
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    _model.to(_device).eval()


def embed_image(path: str) -> list[float]:
    _load()
    img = _preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].tolist()


def embed_text(text: str) -> list[float]:
    _load()
    tokens = _tokenizer([text]).to(_device)
    with torch.no_grad():
        feat = _model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].tolist()
