import os
import re
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import LlavaForConditionalGeneration, AutoProcessor


from transformers import BitsAndBytesConfig



# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# Basic ops
# =====================================================
def l2_normalize(x):
    return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

# =====================================================
# ---------- RETRIEVAL MEMORY -------------------------
# =====================================================
def build_retrieval_memory(npz_root):
    image_bank, caption_bank = [], []
    files = [f for f in os.listdir(npz_root) if f.endswith(".npz")]
    for fname in tqdm(files, desc="Building retrieval bank"):
        data = np.load(os.path.join(npz_root, fname), allow_pickle=True)
        text = data.get("phrase_embeddings", data.get("caption_embedding"))
        if text.ndim == 1:
            text = text.reshape(1, -1)
        patch = data["patch_embeddings"]
        image_bank.append(patch.mean(axis=0))
        caption_bank.append(text.mean(axis=0))
    image_bank = l2_normalize(torch.tensor(np.stack(image_bank), dtype=torch.float32, device=device))
    caption_bank = l2_normalize(torch.tensor(np.stack(caption_bank), dtype=torch.float32, device=device))
    return image_bank, caption_bank

# =====================================================
# Retrieval consistency VECTOR
# =====================================================
def retrieval_consistency_vector(img_vec, cap_vec, image_bank, caption_bank, k=10, eps=1e-8):
    img_vec = l2_normalize(img_vec.unsqueeze(0))
    cap_vec = l2_normalize(cap_vec.unsqueeze(0))

    sim_img = torch.matmul(img_vec, image_bank.T)[0]
    topk_img_idx = torch.topk(sim_img, k=k).indices
    retrieved_caps = caption_bank[topk_img_idx]
    sim_caps = torch.matmul(cap_vec, retrieved_caps.T)[0]

    sim_cap2img = torch.matmul(cap_vec, caption_bank.T)[0]
    topk_cap_idx = torch.topk(sim_cap2img, k=k).indices
    retrieved_imgs = image_bank[topk_cap_idx]
    sim_imgs = torch.matmul(img_vec, retrieved_imgs.T)[0]

    # OOC features
    mean_sim = sim_caps.mean()
    std_sim = sim_caps.std()
    max_sim = sim_caps.max()
    overlap = len(set(topk_img_idx.tolist()) & set(topk_cap_idx.tolist())) / float(k)
    agreement = torch.tensor(overlap, device=device)
    probs = torch.softmax(sim_caps, dim=0)
    entropy = -(probs * torch.log(probs + eps)).sum()
    img_coherence = torch.mean(torch.matmul(retrieved_imgs, retrieved_imgs.T))
    cap_coherence = torch.mean(torch.matmul(retrieved_caps, retrieved_caps.T))
    cross_gap = torch.abs(sim_caps.mean() - sim_imgs.mean())

    vec = torch.stack([mean_sim, std_sim, max_sim, agreement, entropy, img_coherence, cap_coherence, cross_gap])
    vec = (vec - vec.min()) / (vec.max() - vec.min() + eps)
    return vec

# =====================================================
# Gemma VLM Loading
# =====================================================

MODEL_NAME = "/home/models/llava-hf_llava-1.5-7b-hf"
# MODEL_NAME_ONLINE = f'liuhaotian/llava-v1.5-7b'
# MODEL_NAME = "/home/models/gemma-3-27b-it"
print("Loading Llava VLM...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

processor.patch_size = 14
processor.vision_feature_select_strategy = "default"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # computation precision
#     bnb_4bit_use_double_quant=True,       # better accuracy
#     bnb_4bit_quant_type="nf4"             # nf4 or fp4
# )

# gemma = AutoModelForImageTextToText.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     quantization_config=bnb_config
# )

gemma = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

# gemma = AutoModelForImageTextToText.from_pretrained(
#     MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
# )
gemma = gemma.to(device)
gemma.eval()

# =====================================================
# Gemma helpers
# =====================================================

def ensure_1d_string_array(arr, expected_len):
    arr = np.array(arr, dtype=object)

    # flatten safely
    if arr.ndim == 0:
        arr = [str(arr.item())]
    else:
        arr = [str(x) for x in arr.tolist()]

    # HARD CHECK (no silent bugs)
    if len(arr) != expected_len:
        raise ValueError(
            f"Phrase count mismatch: {len(arr)} vs embeddings {expected_len}"
        )

    return arr

# def ensure_1d_string_array(arr, expected_len):
#     arr = np.array(arr, dtype=object)
#     print (arr)

#     # scalar string
#     if arr.ndim == 0:
#         return [str(arr.item())]

#     # already list-like
#     arr = [str(x) for x in arr]

#     if len(arr) != expected_len:
#         print(
#             f"⚠️ Phrase count mismatch: "
#             f"{len(arr)} phrases vs {expected_len} embeddings"
#         )

#     return arr

# def ensure_1d_string_array(arr, expected_len):
#     arr = np.array(arr, dtype=object)
#     print (arr)
#     if arr.ndim == 0: return [str(arr.item())] * expected_len
#     if arr.ndim == 1 and len(arr) == 1: return [str(arr[0])] * expected_len
#     return [str(x) for x in arr]

def extract_patch_paths(arr, expected_len):
    arr = np.array(arr, dtype=object)
    text = str(arr.item() if arr.ndim == 0 else arr[0])
    paths = re.findall(r"\d+\.\s*(\/.*?\.jpg)", text)
    if len(paths) == 0: raise ValueError("No image paths parsed from VISUAL_PATCHES")
    if len(paths) != expected_len:
        print(f"⚠️ Parsed {len(paths)} patches but expected {expected_len}")
    return paths

def build_gemma_prompt(phrase):
    return f"""
You are evaluating semantic causal relevance between text and an image region.

Phrase:
"{phrase}"

Task:
Estimate two probabilities.

P1 = probability image patch visually represents the phrase
P2 = probability phrase was written after observing the patch

Output format:
<probability1> <probability2>

Rules:
- Two decimal numbers only
- No text, No Symbols, Only Numbers
""".strip()



# def build_gemma_prompt(phrase):
#     return f"""
# You are evaluating semantic causal relevance between text and an image region.

# Image Patch:
# <image>

# Phrase:
# "{phrase}"

# Task:
# Estimate two probabilities.

# P1 = probability image patch visually represents the phrase
# P2 = probability phrase was written after observing the patch

# Output format:
# <probability1> <probability2>

# Rules:
# - Two decimal numbers only
# - No text
# """.strip()


@torch.no_grad()
def query_gemma_pair(prompt_text, image_path, max_new_tokens=10):
    image = Image.open(image_path).convert("RGB")

    # LLaVA-1.5 expects this exact chat format
    conversation = f"USER: <image>\n{prompt_text}\nASSISTANT:"

    inputs = processor(
        text=conversation,
        images=image,
        return_tensors="pt"
    ).to(device)

    outputs = gemma.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )

    text = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    print(text)
    return text
# @torch.no_grad()
# def query_gemma_pair(prompt, image_path, max_new_tokens=10):
#     image = Image.open(image_path).convert("RGB")
#     messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
#     inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(gemma.device)
#     outputs = gemma.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3, top_p=0.9)
#     text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
#     print (text)
#     return text

number_regex = re.compile(r"\d+\.\d+")
def parse_pair_probs(text):
    nums = number_regex.findall(text)
    if len(nums) < 2: return 1e-3, 1e-3
    return float(nums[0]), float(nums[1])

def build_gemma_causal_matrices(data):
    n_text, n_patch = len(data["phrase_embeddings"]), len(data["patch_embeddings"])
    phrases = ensure_1d_string_array(data["phrase_texts"], n_text)
    print (phrases)
    patches = extract_patch_paths(data["gemma_patch_input"], n_patch)
    print (patches)
    C_TP, C_PT = torch.zeros((n_text, n_patch), device=device), torch.zeros((n_patch, n_text), device=device)

    print("Building causal matrices (Gemma VLM)...")
    for i in range(n_text):
        prompt = build_gemma_prompt(phrases[i])
        for j in range(n_patch):
            prob_tp, prob_pt = parse_pair_probs(query_gemma_pair(prompt, patches[j]))
            C_TP[i, j], C_PT[j, i] = prob_tp, prob_pt
    print (C_TP)
    print (C_PT)
    return C_TP, C_PT

# =====================================================
# Augmentation pipeline
# =====================================================
def augment_embeddings_with_parents(embeddings, W_dag, retrieval_vector, top_k=5):
    d, dim = embeddings.shape
    rdim = retrieval_vector.shape[0]
    out = torch.zeros((d, 2*dim + rdim), device=embeddings.device)
    for j in range(d):
        w = W_dag[:, j]
        idxs = torch.argsort(torch.abs(w), descending=True)[:top_k]
        idxs = idxs[w[idxs].abs() > 1e-6]
        parent_vec = torch.zeros(dim, device=embeddings.device)
        if len(idxs) > 0:
            probs = w[idxs] / (torch.sum(torch.abs(w[idxs])) + 1e-8)
            parent_vec = l2_normalize(torch.sum(probs.unsqueeze(1) * embeddings[idxs], dim=0).unsqueeze(0))[0]
        out[j] = torch.cat([embeddings[j], parent_vec, retrieval_vector], dim=0)
    return out

# =====================================================
# Load embeddings
# =====================================================
def load_embeddings_from_npz(path):
    data = np.load(path, allow_pickle=True)
    text, patch = data.get("phrase_embeddings", data.get("caption_embedding")), data["patch_embeddings"]
    if text.ndim == 1: text = text.reshape(1, -1)
    text, patch = l2_normalize(torch.from_numpy(text).float().to(device)), l2_normalize(torch.from_numpy(patch).float().to(device))
    return text, patch, data

# =====================================================
# Asymmetric prior
# =====================================================
def build_asymmetric_prior(C_TP, C_PT, text_embs, patch_embs, topk=10, temp=0.9):
    n_text, n_patch = C_TP.shape
    d = n_text + n_patch
    Wp = torch.zeros((d, d), device=device)
    
    for i in range(n_text):
        pos = torch.argsort(C_TP[i], descending=True)[:topk]
        for p in pos:
            j = n_text + p
            Wp[i, j] = torch.tanh(C_TP[i, p] / temp)

    for p in range(n_patch):
        pos = torch.argsort(C_PT[p], descending=True)[:topk]
        for i in pos:
            j = n_text + p
            Wp[j, i] = 0.2 * torch.tanh(C_PT[p, i] / temp)

    Wp.fill_diagonal_(0)
    return Wp


# =====================================================
# DAG projection
# =====================================================
def project_to_acyclic(W0, lr=1e-3, lam=5.0, lam_relevance=1.0, max_iter=500, tol=1e-7):
    W = W0.clone()
    d = W.shape[0]
    
    for step in range(max_iter):
        W_sq = W * W
        exp_W_sq = torch.matrix_exp(W_sq)
        h = torch.trace(exp_W_sq) - d
        if h.item() < tol:
            break
        grad_h = exp_W_sq.T * (2 * W)
        grad = lam * grad_h + lam_relevance * 2 * (W - W0)
        W = W - lr * grad
        W.fill_diagonal_(0)
    return W


# =====================================================
# Top-K DAG
# =====================================================
def topk_parents_dag(W, k=10):
    d = W.shape[0]
    W_dag = torch.zeros_like(W)
    for j in range(d):
        idxs = torch.argsort(torch.abs(W[:, j]), descending=True)[:k]
        for i in idxs:
            if i != j:
                W_dag[i, j] = W[i, j]
    return W_dag

# =====================================================
# MAIN PROCESS
# =====================================================
def process_npz_folder(npz_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_bank, caption_bank = build_retrieval_memory(npz_root)
    files = [f for f in os.listdir(npz_root) if f.endswith(".npz")]

    for fname in tqdm(files):
        path = os.path.join(npz_root, fname)
        text_embs, patch_embs, data = load_embeddings_from_npz(path)
        img_vec, cap_vec = patch_embs.mean(dim=0), text_embs.mean(dim=0)
        r_vec = retrieval_consistency_vector(img_vec, cap_vec, image_bank, caption_bank)

        # ⭐ Use Gemma VLM matrices
        C_TP, C_PT = build_gemma_causal_matrices(data)

        Wp = build_asymmetric_prior(C_TP, C_PT, text_embs, patch_embs)
        W_acyclic = project_to_acyclic(Wp)
        W_dag = topk_parents_dag(W_acyclic)

        print (W_dag)

        all_emb = torch.cat([text_embs, patch_embs], dim=0)
        aug = augment_embeddings_with_parents(all_emb, W_dag, r_vec).cpu().numpy()

        n_text = text_embs.shape[0]
        save_dict = {k: data[k] for k in data.files}
        save_dict.update({
            "augmented_phrase_embeddings": aug[:n_text].astype(np.float32),
            "augmented_patch_embeddings": aug[n_text:].astype(np.float32),
            "retrieval_consistency_vector": r_vec.cpu().numpy().astype(np.float32),
            "retrieval_consistency_score": float(r_vec.mean().cpu())
        })

        np.savez_compressed(os.path.join(out_dir, fname), **save_dict)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    process_npz_folder("clip_amz_test_gemma", "clip_amz_test_gemma_aug")
    print("Done")