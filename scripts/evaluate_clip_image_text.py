#!/usr/bin/env python3
"""用 CLIP（ViT-B/32）计算「图像 ↔ 类别文本」相似度，用于筛查语义可疑样本。

依赖（在 backtool 等已装 torch 的环境）:
  pip install transformers pillow

示例:
  cd /workspace/backdoor-toolbox-new1
  python scripts/evaluate_clip_image_text.py \\
    --images-root /workspace/data/tiny-target-domain-fullshot-v3/images \\
    --class-map /workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json \\
    --output-csv /workspace/data/tiny-target-domain-fullshot-v3/review_outputs/clip_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


def load_folder_to_display_name(class_map_path: Optional[Path]) -> Dict[str, str]:
    """class_dir_name（文件夹名） -> Tiny-ImageNet 显示名（空格分隔）。"""
    if not class_map_path or not class_map_path.is_file():
        return {}
    data = json.loads(class_map_path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for c in data.get("classes", []):
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        key = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip()).strip("_")
        key = re.sub(r"_+", "_", key)
        if key:
            out[key] = name
        wnid = str(c.get("wnid", "")).strip()
        if wnid:
            out[wnid] = name
    return out


def folder_to_label(folder_name: str, name_table: Dict[str, str]) -> str:
    if folder_name in name_table:
        return name_table[folder_name]
    return folder_name.replace("_", " ")


def build_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def clip_cosine(
    model,
    processor,
    images: List[Image.Image],
    texts: List[str],
    device: str,
) -> List[float]:
    """返回每张图与其对应文本的余弦相似度（已归一化 embedding 的点积），形状与 images 一致。"""
    if len(images) != len(texts):
        raise ValueError("images and texts length mismatch")

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 优先走 forward，拿到 image_embeds / text_embeds（新版本最稳）
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        return_dict=True,
    )
    if getattr(out, "image_embeds", None) is not None and getattr(out, "text_embeds", None) is not None:
        image_feats = out.image_embeds.float()
        text_feats = out.text_embeds.float()
    else:
        # 旧版 transformers 或未填充 embeds 时：手写与 CLIPModel.forward 一致的投影
        vout = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = getattr(vout, "pooler_output", None)
        if pooled is None and hasattr(vout, "__getitem__"):
            pooled = vout[1]
        image_feats = model.visual_projection(pooled).float()

        tout = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        tpooled = getattr(tout, "pooler_output", None)
        if tpooled is None and hasattr(tout, "__getitem__"):
            tpooled = tout[1]
        text_feats = model.text_projection(tpooled).float()

    image_feats = image_feats / (image_feats.norm(dim=-1, keepdim=True) + 1e-12)
    text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (image_feats * text_feats).sum(dim=-1)
    return [float(x) for x in sim.cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP image-text similarity per labeled image.")
    parser.add_argument("--images-root", type=str, required=True, help="ImageFolder 根目录，子目录为类名")
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="可选：class_map_tiny_imagenet_200.json，用于文件夹名 -> 显示类名",
    )
    parser.add_argument(
        "--text-template",
        type=str,
        default="a photo of a {}",
        help="文本模板，{} 替换为类显示名",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face 上的 CLIP 模型",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少张图（0 表示不限制，用于试跑）",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_root = Path(args.images_root).resolve()
    if not images_root.is_dir():
        print(f"[ERROR] 目录不存在: {images_root}", file=sys.stderr)
        sys.exit(1)

    class_map_path = Path(args.class_map).resolve() if args.class_map else None
    name_table = load_folder_to_display_name(class_map_path)

    rows: List[Tuple[str, str, str, float]] = []
    all_paths: List[Path] = []
    for cls_dir in sorted([d for d in images_root.iterdir() if d.is_dir()]):
        for p in sorted(cls_dir.glob("*.png")):
            all_paths.append(p)
    if args.limit and args.limit > 0:
        all_paths = all_paths[: args.limit]

    if not all_paths:
        print("[ERROR] 未找到任何 png", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] device={device} model={args.model_id} images={len(all_paths)}")

    model, processor = build_clip(args.model_id, device)

    bs = max(1, args.batch_size)
    for start in range(0, len(all_paths), bs):
        batch_paths = all_paths[start : start + bs]
        imgs: List[Image.Image] = []
        texts: List[str] = []
        metas: List[Tuple[str, str]] = []
        for p in batch_paths:
            cls_folder = p.parent.name
            label = folder_to_label(cls_folder, name_table)
            text = args.text_template.format(label)
            imgs.append(Image.open(p).convert("RGB"))
            texts.append(text)
            metas.append((str(p.resolve()), cls_folder))

        try:
            sc = clip_cosine(model, processor, imgs, texts, device)
        finally:
            for im in imgs:
                im.close()

        for (path_s, folder), s in zip(metas, sc):
            label = folder_to_label(folder, name_table)
            text = args.text_template.format(label)
            rows.append((path_s, folder, text, s))

    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_folder", "text_prompt", "clip_cosine"])
        for r in sorted(rows, key=lambda x: x[3]):
            w.writerow(r)

    scores = [r[3] for r in rows]
    print(f"[OK] wrote {out_path}")
    print(
        f"[STATS] min={min(scores):.4f} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}"
    )
    worst = sorted(rows, key=lambda x: x[3])[:15]
    print("[WORST 15] (low cosine -> 更可疑)")
    for path_s, folder, text, s in worst:
        print(f"  {s:.4f}  {folder}  {Path(path_s).name}")


if __name__ == "__main__":
    main()
