import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import os, json, re
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


# =====================================================
# NER MODEL
# =====================================================
ner_model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER",
    local_files_only=True,
)

ner_tokenizer = AutoTokenizer.from_pretrained(
    "dslim/bert-base-NER",
    local_files_only=True,
)

# =====================================================
# PATHS
# =====================================================
TRAIN_JSON = "/home/arghodeep/rel/AMG_json/val.json"
PATCHES_ROOT = "/home/arghodeep/rel/AMG_GRID_VAL"
OUTPUT_ROOT = "clip_amz_val_gemma"

DEVICE_PREF = "auto"
TEXT_BATCH = 256
IMAGE_BATCH = 256
NER_BATCH = 32
LIMIT = None
TOP_K_PATCHES = 15


# =====================================================
# DEVICE
# =====================================================
def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = pick_device(DEVICE_PREF)


# =====================================================
# CLIP ENCODER
# =====================================================
class CLIPEncoderHF:
    def __init__(self, device: torch.device):

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=True
        ).to(device)

        if device.type == "cuda":
            self.model = self.model.half()

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            use_fast=True,
            local_files_only=True
        )

        self.device = device
        self.embed_dim = 768
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: List[str]):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feats = self.model.get_text_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats

    @torch.no_grad()
    def encode_images(self, images):

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feats = self.model.get_image_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats


# =====================================================
# IMAGE UTILS
# =====================================================
def list_patch_images(patch_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not patch_dir.exists():
        return []
    return sorted(
        p for p in patch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# =====================================================
# TEXT CLEANING
# =====================================================
URL_PATTERN = re.compile(r"https?://\S+")


def clean_caption(text: str):
    text = URL_PATTERN.sub("", text)
    return text.strip()


_punct = r"""[.,;:!?\\"'()\[\]\{\}]"""


def strip_punct(s: str):
    return re.sub(rf"^{_punct}|{_punct}$", "", s)


# =====================================================
# GEMMA FORMATTERS (UPDATED)
# =====================================================
def format_gemma_phrases(caption: str, phrases: List[str]):

    lines = ["[CAPTION]", caption, "", "[PHRASES]"]

    for i, p in enumerate(phrases, 1):
        lines.append(f"{i}. {p}")

    return "\n".join(lines)


def format_gemma_patches(patch_paths: List[Path]):
    """
    Uses actual JPG filenames (NO similarity scores).
    """

    lines = ["[VISUAL_PATCHES]"]

    for i, p in enumerate(patch_paths, 1):
        lines.append(f"{i}. {p.as_posix()}")

    return "\n".join(lines)


# =====================================================
# PHRASE EXTRACTION
# =====================================================
def build_phrases_batch(captions, ner_pipe):

    ner_out = ner_pipe(captions, batch_size=NER_BATCH)
    all_phrases = []

    for caption, entities in zip(captions, ner_out):

        spans = sorted((int(e["start"]), int(e["end"])) for e in entities)

        phrases = []
        i = si = 0
        n = len(caption)

        while i < n:

            if caption[i].isspace():
                i += 1
                continue

            if si < len(spans) and i == spans[si][0]:
                s, t = spans[si]
                p = strip_punct(caption[s:t])
                if p:
                    phrases.append(p)
                i = t
                si += 1
            else:
                j = i
                while j < n and not caption[j].isspace():
                    j += 1

                p = strip_punct(caption[i:j])
                if p:
                    phrases.append(p)

                i = j

        all_phrases.append(phrases)

    return all_phrases


# =====================================================
# LOAD DATA
# =====================================================
def load_records(path: str):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# =====================================================
# MAIN
# =====================================================
def main():

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoderHF(device)

    ner_pipe = pipeline(
        "ner",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=0 if device.type == "cuda" else -1,
    )

    records = load_records(TRAIN_JSON)

    if LIMIT is not None:
        records = records[:LIMIT]

    captions = [clean_caption(rec.get("content", "")) for rec in records]

    phrases_per_caption = build_phrases_batch(captions, ner_pipe)

    # ---------------- flatten texts ----------------
    flat_texts = []
    phrase_ranges = []

    for cap, phr in zip(captions, phrases_per_caption):

        cap_idx = len(flat_texts)
        flat_texts.append(cap)

        start = len(flat_texts)
        flat_texts.extend(phr)
        end = len(flat_texts)

        phrase_ranges.append((cap_idx, start, end))

    # ---------------- encode texts ----------------
    text_embeddings = []

    for i in range(0, len(flat_texts), TEXT_BATCH):
        feats = encoder.encode_text(flat_texts[i:i + TEXT_BATCH])
        text_embeddings.append(feats)

    text_embeddings = torch.cat(text_embeddings).cpu().numpy().astype(np.float32)

    # =====================================================
    # PER SAMPLE PROCESSING
    # =====================================================
    saved = 0

    for idx, rec in enumerate(tqdm(records, desc="Encoding")):

        image_id = str(rec["Id"])
        sample_label = str(rec.get("label", ""))

        cap_i, p_s, p_e = phrase_ranges[idx]

        caption_vec = text_embeddings[cap_i:cap_i + 1]
        phrase_vecs = text_embeddings[p_s:p_e]
        phrase_texts = phrases_per_caption[idx]

        patch_dir = Path(PATCHES_ROOT) / image_id
        patch_files = list_patch_images(patch_dir)

        if not patch_files:
            continue

        with ThreadPoolExecutor(max_workers=8) as ex:
            imgs = list(filter(None, ex.map(load_image, patch_files)))

        if not imgs:
            continue

        patch_embs = []

        for i in range(0, len(imgs), IMAGE_BATCH):
            feats = encoder.encode_images(imgs[i:i + IMAGE_BATCH])
            patch_embs.append(feats)

        patch_matrix = torch.cat(patch_embs, dim=0)

        caption_t = torch.from_numpy(caption_vec).to(
            device=patch_matrix.device,
            dtype=patch_matrix.dtype,
        )

        sims = (patch_matrix @ caption_t.T).squeeze(1)

        k = min(TOP_K_PATCHES, sims.numel())
        _, topk_idx = torch.topk(sims, k=k, largest=True)

        topk_list = topk_idx.cpu().tolist()

        selected_patch_files = [patch_files[i] for i in topk_list]
        selected_patch_embs = patch_matrix[topk_list].cpu().numpy().astype(np.float32)

        # selected_patch_embs = (
        #     patch_matrix[topk_idx].cpu().numpy().astype(np.float32)
        # )

        # # ===== NEW: select filenames =====
        # selected_patch_files = [
        #     patch_files[i] for i in topk_idx.cpu().tolist()
        # ]

        # =====================================================
        # GEMMA INPUTS
        # =====================================================
        gemma_phrase_input = format_gemma_phrases(
            captions[idx],
            phrase_texts,
        )

        gemma_patch_input = format_gemma_patches(
            selected_patch_files
        )

        # =====================================================
        # SAVE
        # =====================================================
        out_path = Path(OUTPUT_ROOT) / f"{image_id}.npz"

        np.savez_compressed(
            out_path,
            caption_embedding=caption_vec,
            phrase_embeddings=phrase_vecs,
            phrase_texts=np.array(phrase_texts, dtype=object),

            # ordering preserved (same indices as filenames)
            patch_embeddings=selected_patch_embs,

            image_id=np.array([image_id], dtype=object),
            id=np.array([sample_label], dtype=object),

            # IMPORTANT: avoid 0-D arrays
            gemma_phrase_input=np.array([gemma_phrase_input], dtype=object),
            gemma_patch_input=np.array([gemma_patch_input], dtype=object),
        )

        saved += 1

    print(f"[Done] Saved {saved} NPZ files → {OUTPUT_ROOT}")


# =====================================================
if __name__ == "__main__":
    main()