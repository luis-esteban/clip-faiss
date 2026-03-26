from faiss import read_index
import clip
import os


def _read_text_with_fallback(path):
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            with open(path, encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")
import torch


class App:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        self.index = read_index("static/index.faiss")
        urls_path = os.path.join("static", "dal_urls.txt")
        text = _read_text_with_fallback(urls_path)
        self.image_urls = [line.strip() for line in text.splitlines() if line.strip()]

    def search(self, search_text, results=1):
        text_tokens = clip.tokenize([search_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

        _, indices = self.index.search(text_features, results)
        return [self.image_urls[indices[0][i]] for i in range(results)]

    def run(self):
        while True:
            search_text = input("Search: ")
            if search_text == "exit":
                break
            image_url = self.search(search_text)[0]
            print(image_url)


if __name__ == "__main__":
    app = App()
    app.run()
