from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import os
import json
from tqdm import tqdm

# =====================================================
# Utils
# =====================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


# =====================================================
# Patch Saving
# =====================================================

def save_tile_worker(args):
    img, x1, y1, x2, y2, out_path, quality = args

    tile = img[y1:y2, x1:x2]

    cv2.imwrite(
        out_path,
        tile,
        [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
        ],
    )


def generate_tasks(img, base, output_dir, grid_sizes, overlap_ratio, fmt, quality):

    H, W, _ = img.shape
    idx = 0

    for grid in grid_sizes:

        cell_w = W // grid
        cell_h = H // grid

        stride_w = max(1, int(cell_w * (1 - overlap_ratio)))
        stride_h = max(1, int(cell_h * (1 - overlap_ratio)))

        row_idx = 0
        y = 0
        while y <= max(0, H - cell_h):

            col_idx = 0
            x = 0
            while x <= max(0, W - cell_w):

                x1, y1 = x, y
                x2, y2 = x + cell_w, y + cell_h

                out_name = f"{base}_g{grid}_r{row_idx}_c{col_idx}_{idx}.{fmt}"
                out_path = os.path.join(output_dir, out_name)

                yield (img, x1, y1, x2, y2, out_path, quality)

                idx += 1
                col_idx += 1
                x += stride_w

            row_idx += 1
            y += stride_h

    out_name = f"{base}_full_{idx}.{fmt}"
    yield (img, 0, 0, W, H, os.path.join(output_dir, out_name), quality)


def save_multiscale_patches(
        img,
        base,
        output_dir,
        grid_sizes,
        overlap_ratio,
        fmt,
        quality,
        num_workers=os.cpu_count(),
):

    ensure_dir(output_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        ex.map(
            save_tile_worker,
            generate_tasks(img, base, output_dir,
                           grid_sizes, overlap_ratio, fmt, quality),
            chunksize=16,
        )


# =====================================================
# Video → Frames
# =====================================================

def extract_video_frames(video_path, max_frames=5):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    step = max(1, total // max_frames)

    frames = []
    idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        idx += step

        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


# =====================================================
# Resolve media path (SINGLE FOLDER VERSION)
# =====================================================

def resolve_media_path(media_id, media_folder):

    stem = os.path.splitext(str(media_id))[0]

    for ext in IMAGE_EXTS + VIDEO_EXTS:
        candidate = os.path.join(media_folder, stem + ext)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return None


# =====================================================
# Output Folder
# =====================================================

def build_output_dir(media_id, patches_root):

    stem = Path(media_id).stem
    out_dir = os.path.join(patches_root, "origin", stem)
    ensure_dir(out_dir)
    return out_dir


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    JSON_PATH = "/home/arghodeep/rel/AMG_json/train.json"
    MEDIA_FOLDER = "/home/arghodeep/rel/AMG_MEDIA/AMG_MEDIA/train"
    PATCHES_ROOT = "/home/arghodeep/rel/AMG_GRID_TRAIN"

    GRID_SIZES = [3, 4]
    OVERLAP_RATIO = 0.1
    JPEG_QUALITY = 95
    FMT = "jpg"

    ensure_dir(PATCHES_ROOT)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in tqdm(records, desc="Extracting Grid Subimages"):

        media_id = rec.get("Id")
        if not media_id:
            continue

        abs_path = resolve_media_path(media_id, MEDIA_FOLDER)

        if abs_path is None:
            print(f"Missing media for ID: {media_id}")
            continue

        out_dir = build_output_dir(media_id, PATCHES_ROOT)

        ext = Path(abs_path).suffix.lower()

        try:

            # ---------------- IMAGE ----------------
            if ext in IMAGE_EXTS:

                img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Image read failed")

                save_multiscale_patches(
                    img,
                    Path(abs_path).stem,
                    out_dir,
                    GRID_SIZES,
                    OVERLAP_RATIO,
                    FMT,
                    JPEG_QUALITY,
                )

            # ---------------- VIDEO ----------------
            elif ext in VIDEO_EXTS:

                frames = extract_video_frames(abs_path, max_frames=5)

                for i, frame in enumerate(frames):

                    base = f"{Path(abs_path).stem}_frame{i}"

                    save_multiscale_patches(
                        frame,
                        base,
                        out_dir,
                        GRID_SIZES,
                        OVERLAP_RATIO,
                        FMT,
                        JPEG_QUALITY,
                    )

        except Exception as e:
            print(f"Error processing {abs_path}: {e}")

    print("\nGrid Subimages written under:", PATCHES_ROOT)