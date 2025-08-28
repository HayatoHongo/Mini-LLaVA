# --- drop-in replacement: no HF datasets, no trust_remote_code ---
import os, json, tarfile, io, shutil, requests
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass

# あなたの .config 側の定義をそのまま利用
from .config import DataArguments

DOCCI_BASE = "https://storage.googleapis.com/docci/data/"
DOCCI_DESC_URL = DOCCI_BASE + "docci_descriptions.jsonlines"
DOCCI_IMG_TAR_URL = DOCCI_BASE + "docci_images.tar.gz"

def _stream_download(url: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return dst_path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(dst_path)}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dst_path

def _safe_extract_tar(tar_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            target = os.path.abspath(os.path.join(extract_dir, m.name))
            if not target.startswith(os.path.abspath(extract_dir) + os.sep):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(extract_dir)

def prepare_docci_data(output_json_path, image_folder="data/docci", split="train", reuse_images=True):
    """
    HF datasets を使わず DOCCI をローカルに変換。
    split: 'train' | 'test' | 'qual_dev' | 'qual_test'
    reuse_images=True なら既保存画像があれば再利用（再変換を早くする）
    """
    os.makedirs(image_folder, exist_ok=True)
    cache_dir = os.path.join(image_folder, "_raw")
    os.makedirs(cache_dir, exist_ok=True)

    # 1) ダウンロード
    desc_path = _stream_download(DOCCI_DESC_URL, os.path.join(cache_dir, "docci_descriptions.jsonlines"))
    tar_path  = _stream_download(DOCCI_IMG_TAR_URL,  os.path.join(cache_dir, "docci_images.tar.gz"))

    # 2) 展開（tar の中に images/ が入っている）
    extracted_dir = os.path.join(cache_dir, "extracted")
    if not os.path.isdir(os.path.join(extracted_dir, "images")):
        _safe_extract_tar(tar_path, extracted_dir)
    images_root = os.path.join(extracted_dir, "images")

    # 3) descriptions を読んで split ごとに画像ファイルへマップ
    def _keep(ex):
        # 公式スクリプトと同じ判定（split と example_id 前置きの両方）に合わせる
        prefix = split
        return (ex.get("split") == split) and ex.get("example_id", "").startswith(prefix)

    data_json = []
    idx = 0
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Converting ({split})"):
            ex = json.loads(line)
            if not _keep(ex):
                continue
            src_path = os.path.join(images_root, ex["image_file"])

            # 保存先ファイル名はあなたの元コードに合わせて連番化
            img_filename = f"docci_{idx}.jpg"
            dst_path = os.path.join(image_folder, img_filename)

            if not (reuse_images and os.path.exists(dst_path)):
                # 画像はRGBで再保存（壊れ画像対策で例外時はコピーにフォールバック）
                try:
                    Image.open(src_path).convert("RGB").save(dst_path, quality=95)
                except Exception:
                    shutil.copy2(src_path, dst_path)

            data_json.append({
                "id": f"docci_{idx}",
                "media": [{"image": img_filename}],
                "conversations": [
                    {"from": "human", "value": "<image>\nCan you describe this image?"},
                    {"from": "gpt",   "value": ex["description"]}
                ]
            })
            idx += 1

    # 4) JSON 出力
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    # 5) あなたの DataArguments で返す（元実装と同じフィールド）
    data_args = DataArguments(
        data_path = output_json_path,
        image_folder = image_folder + "/",
        video_folder = image_folder + "/",
        video_fps = 1,
        frames_upbound = 0,
        add_time_instruction = False,
        force_sample = False,
        default_fps = 10,
    )
    return data_args
