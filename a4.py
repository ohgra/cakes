import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel
from tqdm import tqdm

# =====================================================
# CONFIG (MATCH TRAINING)
# =====================================================

MAX_POS = 512
MODEL_PATH = "t5_llava6e5_15ep_768p.pt"
EMBEDDINGS_ROOT = "clip_amz_test_gemma_768_p_aug"
BATCH_SIZE = 64

CLIP_DIM = 768      # MUST MATCH TRAIN
# ab1=1536
# ab2=776
# ab3=768
# original=1544
NUM_CLASSES = 6

LABEL2ID = {str(i): i for i in range(NUM_CLASSES)}
ID2LABEL = {i: str(i) for i in range(NUM_CLASSES)}

# =====================================================
# DATASET
# =====================================================

class CLIPNerPatchDataset(Dataset):
    def __init__(self, root: str):
        self.items = []

        for f in sorted(os.listdir(root)):
            if not f.endswith(".npz"):
                continue

            path = os.path.join(root, f)

            try:
                data = np.load(path, allow_pickle=True)
            except Exception:
                continue

            # -------- SAME LABEL LOGIC AS TRAIN --------
            raw_id = data.get("id", 0)

            try:
                raw_id = raw_id.item() if hasattr(raw_id, "item") else raw_id
                label = int(raw_id)
            except Exception:
                label = 0

            label = max(0, min(label, NUM_CLASSES - 1))
            # -------------------------------------------

            self.items.append((path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data = np.load(path, allow_pickle=True)

        phrase = data["augmented_phrase_embeddings"].astype(np.float32)
        patch = data["augmented_patch_embeddings"].astype(np.float32)

        phrase = np.nan_to_num(phrase, nan=0.0, posinf=1e6, neginf=-1e6)
        patch = np.nan_to_num(patch, nan=0.0, posinf=1e6, neginf=-1e6)

        seqs = []

        if phrase.shape[0] > 0:
            seqs.append(phrase)

        if patch.shape[0] > 0:
            seqs.append(patch)

        if len(seqs) == 0:
            seq = np.zeros((1, CLIP_DIM), dtype=np.float32)
        else:
            seq = np.vstack(seqs)

        return torch.from_numpy(seq), label


# =====================================================
# COLLATE
# =====================================================

def pad_collate(batch, pad_value: float = 0.0, max_cap: int = MAX_POS):
    seqs, labels = zip(*batch)

    max_len = min(max(s.shape[0] for s in seqs), max_cap)
    dim = seqs[0].shape[1]

    xs, masks = [], []

    for s in seqs:
        s = s[:max_len]
        cur_len = s.shape[0]
        pad_len = max_len - cur_len

        if pad_len > 0:
            pad = torch.full((pad_len, dim), pad_value, dtype=s.dtype)
            s2 = torch.cat([s, pad], dim=0)
        else:
            s2 = s

        mask = torch.zeros(max_len, dtype=torch.long)
        mask[:cur_len] = 1

        xs.append(s2)
        masks.append(mask)

    return (
        torch.stack(xs),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.long),
    )


# =====================================================
# MODEL (IDENTICAL TO TRAIN)
# =====================================================

class T5CLIPClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()

        #self.encoder = T5EncoderModel.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained(
            "t5-base",
            local_files_only=True
        )
        d_model = self.encoder.config.d_model

        self.input_proj = nn.Linear(CLIP_DIM, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, mask):
        x = self.input_proj(x)

        B, L, D = x.shape
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)

        attn_mask = torch.cat(
            [torch.ones((B, 1), device=x.device, dtype=torch.long), mask],
            dim=1,
        )

        out = self.encoder(inputs_embeds=x, attention_mask=attn_mask)
        cls_out = out.last_hidden_state[:, 0]

        return self.head(self.norm(cls_out))


# =====================================================
# EVALUATION
# =====================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CLIPNerPatchDataset(EMBEDDINGS_ROOT)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
    )

    model = T5CLIPClassifier().to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    total, correct = 0, 0

    with torch.no_grad():
        for x, mask, y in tqdm(loader, desc="Evaluating"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            logits = model(x, mask)
            preds = logits.argmax(-1)

            correct += (preds == y).sum().item()
            total += x.size(0)

            for yi, pi in zip(y.view(-1), preds.view(-1)):
                conf[yi, pi] += 1

    conf = conf.cpu().numpy()
    acc = correct / max(total, 1)

    print("\nAccuracy:", f"{acc:.4f}\n")

    macro_p, macro_r, macro_f1 = [], [], []

    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = (2 * p * r) / max(p + r, 1e-6)

        macro_p.append(p)
        macro_r.append(r)
        macro_f1.append(f1)

        print(f"Class {ID2LABEL[c]}:")
        print(f" Precision: {p:.4f}")
        print(f" Recall:    {r:.4f}")
        print(f" F1:        {f1:.4f}\n")

    print("Macro Precision:", f"{np.mean(macro_p):.4f}")
    print("Macro Recall:", f"{np.mean(macro_r):.4f}")
    print("Macro F1:", f"{np.mean(macro_f1):.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(conf)


if __name__ == "__main__":
    main()