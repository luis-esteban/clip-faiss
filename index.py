from faiss import write_index
from PIL import Image
from tqdm import tqdm

import argparse
import clip
import faiss
import io
import json
import numpy as np
import os
import requests
import torch


def _load_image_from_url(url, timeout=10):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _resolve_input_path(path):
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def _read_text_with_fallback(path):
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            with open(path, encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def _load_urls_from_text(path):
    text = _read_text_with_fallback(path)
    return [line.strip() for line in text.splitlines() if line.strip()]


def index(urls_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    images = []
    image_urls = []
    urls_path = _resolve_input_path(urls_path)
    urls = _load_urls_from_text(urls_path)
    if not urls:
        raise ValueError(f"URL list file is empty: {urls_path}")

    for url in tqdm(urls):
        try:
            image = _load_image_from_url(url)
        except Exception as exc:
            print(f"Skipping {url}: {exc}")
            continue
        images.append(preprocess(image))
        image_urls.append(url)

    if not images:
        raise ValueError("No images were loaded from the provided URLs.")
    image_input = torch.tensor(np.stack(images)).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()

    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "static/index.faiss")

    with open("static/image_paths.json", "w") as f:
        json.dump(image_urls, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=str, default="dal_urls.txt")
    args = parser.parse_args()
    index(args.urls)
