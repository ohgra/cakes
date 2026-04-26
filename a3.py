import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel
from tqdm import tqdm
from torch import amp

# =====================================================
# CONFIG
# =====================================================

EMBEDDINGS_ROOT = "clip_amz_train_gemma_768_np_aug"
VAL_ROOT = "clip_amz_val_gemma_768_np_aug"
SAVE_PATH = "t5_llava6e5_15ep_768np.pt"

BATCH_SIZE = 32
EPOCHS = 15
LR = 6e-5
SEED = 42
MAX_POS = 512
CLIP_DIM = 768
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

            # ---------- NEW LABEL LOGIC ----------
            raw_id = data.get("id", 0)

            try:
                raw_id = raw_id.item() if hasattr(raw_id, "item") else raw_id
                label = int(raw_id)
            except Exception:
                label = 0

            label = max(0, min(label, NUM_CLASSES - 1))
            # -------------------------------------

            self.items.append((path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data = np.load(path, allow_pickle=True)

        phrase_embeds = data["augmented_phrase_embeddings"].astype(np.float32)
        patch_embeds = data["augmented_patch_embeddings"].astype(np.float32)

        phrase_embeds = np.nan_to_num(
            phrase_embeds, nan=0.0, posinf=1e6, neginf=-1e6
        )
        patch_embeds = np.nan_to_num(
            patch_embeds, nan=0.0, posinf=1e6, neginf=-1e6
        )

        if phrase_embeds.shape[0] == 0:
            seq = patch_embeds
        elif patch_embeds.shape[0] == 0:
            seq = phrase_embeds
        else:
            seq = np.vstack([phrase_embeds, patch_embeds])

        if seq.shape[0] == 0:
            seq = np.zeros((1, CLIP_DIM), dtype=np.float32)

        return torch.from_numpy(seq), label


# =====================================================
# COLLATE
# =====================================================

def pad_collate(batch, pad_value: float = 0.0, max_cap: int = MAX_POS):
    seqs, labels = zip(*batch)

    max_len = min(max(s.shape[0] for s in seqs), max_cap)
    emb_dim = seqs[0].shape[1]

    padded, masks = [], []

    for s in seqs:
        s = s[:max_len]
        pad_len = max_len - s.shape[0]

        if pad_len > 0:
            s2 = torch.cat(
                [s, torch.full((pad_len, emb_dim), pad_value, dtype=s.dtype)],
                dim=0,
            )
            mask = torch.zeros(max_len, dtype=torch.long)
            mask[: s.shape[0]] = 1
        else:
            s2 = s
            mask = torch.ones(max_len, dtype=torch.long)

        padded.append(s2)
        masks.append(mask)

    return (
        torch.stack(padded),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.long),
    )


# =====================================================
# MODEL
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
# LOSS
# =====================================================

def compute_class_weights(ds):
    counts = np.zeros(NUM_CLASSES)

    for _, label in ds.items:
        counts[label] += 1

    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * ((1 - pt) ** self.gamma) * ce_loss
        else:
            loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =====================================================
# TRAIN
# =====================================================

def train_one_epoch(model, loader, optim, device, scaler, criterion):
    model.train()

    total_loss, correct, total = 0, 0, 0

    for x, mask, y in tqdm(loader, desc="Train", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optim.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=False):
            logits = model(x, mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss, correct, total = 0, 0, 0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    for x, mask, y in tqdm(loader, desc="Val", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        with amp.autocast(device_type=device.type, enabled=False):
            logits = model(x, mask)
            loss = criterion(logits, y)

        preds = logits.argmax(-1)

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)

        for yi, pi in zip(y.view(-1), preds.view(-1)):
            conf[yi, pi] += 1

    f1s = []

    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

        f1s.append((2 * prec * rec) / max(prec + rec, 1e-6))

    return total_loss / total, correct / total, float(np.mean(f1s))


# =====================================================
# MAIN
# =====================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CLIPNerPatchDataset(EMBEDDINGS_ROOT)
    val_ds = CLIPNerPatchDataset(VAL_ROOT)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )

    model = T5CLIPClassifier().to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0

    for e in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optim, device, scaler, criterion
        )

        va_loss, va_acc, va_f1 = evaluate(
            model, val_loader, device, criterion
        )

        print(
            f"Epoch {e:02d} "
            f"Train Loss {tr_loss:.4f} Acc {tr_acc:.3f} "
            f"Val Loss {va_loss:.4f} Acc {va_acc:.3f} F1 {va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": LABEL2ID,
                    "val_f1": va_f1,
                },
                SAVE_PATH,
            )

            print(f"Saved best model (F1={va_f1:.4f})")

    print(f"\n[Done] Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()