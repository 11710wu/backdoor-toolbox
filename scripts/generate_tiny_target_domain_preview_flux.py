#!/usr/bin/env python3
"""Generate Tiny target-domain preview set (200 classes x 1 image) with FLUX.1 schnell.

This script focuses on the preview stage:
- one image per Tiny-ImageNet class
- unified photo-realistic style anchor
- reproducible seeds and per-class manifest
"""

import argparse
import json
import os
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps, ImageDraw


STYLE_ANCHOR = "photorealistic DSLR, natural light, realistic texture, sharp focus"

NEGATIVE_PROMPT = (
    "illustration, cartoon, anime, painting, sketch, comic, digital art, "
    "3d render, cgi, unreal engine, game asset, plastic texture, waxy skin, "
    "blurry, low resolution, low quality, over-smoothed, watermark, text, logo"
)

PROMPT_TEMPLATE = "Photo of {class_name}, {scene_hint}, " + STYLE_ANCHOR + "."


SCENE_HINTS = [
    "centered subject, clean composition",
    "outdoor context with natural perspective",
    "lifelike details with believable background",
    "documentary framing with realistic context",
]


@dataclass
class ClassRecord:
    index: int
    wnid: str
    name: str
    synonyms: List[str]


def sanitize_class_dir_name(name: str) -> str:
    """Convert class display name to a safe folder name."""
    # Keep letters/numbers/_/-, map other chars to underscore.
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_class"


def load_class_map(class_map_path: Path) -> List[ClassRecord]:
    payload = json.loads(class_map_path.read_text(encoding="utf-8"))
    classes = [
        ClassRecord(
            index=int(c["index"]),
            wnid=str(c["wnid"]),
            name=str(c["name"]),
            synonyms=[str(x) for x in c.get("synonyms", [])],
        )
        for c in payload["classes"]
    ]
    classes.sort(key=lambda x: x.index)
    return classes


def choose_scene_hint(seed: int, class_index: int) -> str:
    rng = random.Random(seed + 10007 * class_index)
    return SCENE_HINTS[rng.randrange(len(SCENE_HINTS))]


def build_prompt(class_name: str, scene_hint: str) -> str:
    return PROMPT_TEMPLATE.format(class_name=class_name, scene_hint=scene_hint)


def load_semantic_spec(semantic_spec_path: Optional[Path]) -> Dict[str, Dict]:
    if semantic_spec_path is None:
        return {}
    payload = json.loads(semantic_spec_path.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    table = {}
    for row in classes:
        wnid = str(row.get("wnid", "")).strip()
        if wnid:
            table[wnid] = row
    return table


def build_disambiguated_prompt(
    rec: ClassRecord,
    scene_hint: str,
    semantic_row: Optional[Dict],
    max_include_keywords: int = 3,
) -> str:
    if not semantic_row:
        return build_prompt(rec.name, scene_hint)

    sense = str(semantic_row.get("sense", "")).strip()
    include_keywords = [str(x).strip() for x in semantic_row.get("include_keywords", []) if str(x).strip()]
    scene_override = str(semantic_row.get("scene_hint_override", "")).strip()
    framing_hint = str(semantic_row.get("framing_hint", "")).strip()

    pieces = [f"Photo of {rec.name}"]
    if sense:
        pieces.append(f"specifically {sense}")
    if include_keywords:
        pieces.append("include " + ", ".join(include_keywords[:max_include_keywords]))
    if scene_override:
        pieces.append(scene_override)
    else:
        pieces.append(scene_hint)
    if framing_hint:
        pieces.append(f"framing {framing_hint}")
    pieces.append(STYLE_ANCHOR)
    return ", ".join(pieces) + "."


def build_disambiguated_negative_prompt(semantic_row: Optional[Dict], max_exclude_keywords: int = 3) -> str:
    parts = [NEGATIVE_PROMPT]
    if semantic_row:
        exclude_keywords = [str(x).strip() for x in semantic_row.get("exclude_keywords", []) if str(x).strip()]
        if exclude_keywords:
            parts.append("avoid " + ", ".join(exclude_keywords[:max_exclude_keywords]))
    return ", ".join(parts)


def center_crop_resize_64(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((64, 64), Image.Resampling.LANCZOS)


def make_preview_grid(image_paths: List[Path], output_path: Path, cols: int = 20) -> None:
    if not image_paths:
        return
    thumb = 64
    pad = 4
    label_h = 14
    rows = math.ceil(len(image_paths) / cols)
    canvas_w = cols * (thumb + pad) + pad
    canvas_h = rows * (thumb + label_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, path in enumerate(image_paths):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + label_h + pad)
        img = Image.open(path).convert("RGB")
        canvas.paste(img, (x, y))
        label = path.parent.name
        draw.text((x, y + thumb + 1), label, fill=(220, 220, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def apply_hf_endpoint(hf_endpoint: str | None) -> None:
    """必须在首次 import diffusers/huggingface_hub 之前调用。"""
    if not hf_endpoint:
        return
    url = hf_endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = url
    print(f"[HF] HF_ENDPOINT={url} (镜像端点，用于 from_pretrained 下载)")


def get_flux_pipeline(model_id: str, device: str, dtype: str, offload: str = "auto"):
    import torch
    from diffusers import FluxPipeline  # lazy import

    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    # 对 FLUX 这类大模型，默认在 CUDA 下优先使用 CPU offload，避免 pipe.to(cuda) 直接 OOM
    offload_mode = offload
    if offload_mode == "auto":
        offload_mode = "model" if str(device).startswith("cuda") else "none"

    if offload_mode == "model":
        try:
            pipe.enable_model_cpu_offload()
            print("[Memory] enable_model_cpu_offload enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[Memory] enable_model_cpu_offload failed: {e}; fallback to pipe.to({device})")
            pipe = pipe.to(device)
    elif offload_mode == "sequential":
        try:
            pipe.enable_sequential_cpu_offload()
            print("[Memory] enable_sequential_cpu_offload enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[Memory] enable_sequential_cpu_offload failed: {e}; fallback to pipe.to({device})")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    # 降低显存峰值
    try:
        pipe.enable_attention_slicing()
        print("[Memory] attention_slicing enabled")
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=False)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tiny target-domain preview set with FLUX.")
    parser.add_argument(
        "--class-map",
        type=str,
        default="./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/tiny-target-domain-preview-1shot",
    )
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument(
        "--gen-size",
        type=int,
        default=64,
        help="模型生成分辨率（正方形），最终仍会中心裁剪并缩放到64x64保存",
    )
    parser.add_argument("--base-seed", type=int, default=20260414)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="每个类别生成的样本数量（默认1）",
    )
    parser.add_argument(
        "--offload",
        type=str,
        choices=["auto", "none", "model", "sequential"],
        default="auto",
        help="显存策略：auto(默认), none(直接上卡), model(model_cpu_offload), sequential(sequential_cpu_offload)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--use-hf-mirror",
        action="store_true",
        help="国内镜像：在 import diffusers 前设置 HF_ENDPOINT=https://hf-mirror.com",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="自定义 Hugging Face 端点（如 https://hf-mirror.com），优先于 --use-hf-mirror",
    )
    parser.add_argument(
        "--semantic-spec",
        type=str,
        default=None,
        help="语义消歧义配置文件（JSON，含 wnid/sense/include_keywords/exclude_keywords）",
    )
    parser.add_argument(
        "--strict-disambiguation",
        action="store_true",
        help="严格消歧义：强制将 include/exclude 关键词拼入 prompt 和 negative_prompt",
    )
    parser.add_argument("--max-include-keywords", type=int, default=3, help="每类最多拼接多少个 include 关键词")
    parser.add_argument("--max-exclude-keywords", type=int, default=3, help="每类最多拼接多少个 exclude 关键词")
    args = parser.parse_args()

    # 与「import os; os.environ['HF_ENDPOINT']=...」等价，须在首次加载模型前执行
    if args.hf_endpoint:
        apply_hf_endpoint(args.hf_endpoint)
    elif args.use_hf_mirror:
        apply_hf_endpoint("https://hf-mirror.com")

    class_map_path = Path(args.class_map).resolve()
    out_root = Path(args.output_dir).resolve()
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    classes = load_class_map(class_map_path)
    if len(classes) != 200:
        print(f"[WARN] class count is {len(classes)} (expected 200)")
    semantic_spec_path = Path(args.semantic_spec).resolve() if args.semantic_spec else None
    semantic_table = load_semantic_spec(semantic_spec_path)
    missing_semantics: List[Dict] = []

    if args.strict_disambiguation and semantic_spec_path:
        miss = [rec.wnid for rec in classes if rec.wnid not in semantic_table]
        if miss:
            print(f"[WARN] strict-disambiguation enabled but {len(miss)} classes missing semantic rows")
    elif args.strict_disambiguation and not semantic_spec_path:
        print("[WARN] --strict-disambiguation enabled without --semantic-spec; fallback to base prompt")

    pipe = None
    if not args.dry_run:
        pipe = get_flux_pipeline(args.model_id, args.device, args.dtype, offload=args.offload)

    manifest_path = out_root / "manifest_preview_1shot.jsonl"
    failed_cases_path = out_root / "failed_cases.json"
    missing_semantics_path = out_root / "missing_semantics.json"
    grid_path = out_root / "preview_grid.png"

    image_paths: List[Path] = []
    failed_cases: List[Dict] = []

    used_dir_names: Dict[str, int] = {}

    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1")

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rec in classes:
            base_dir_name = sanitize_class_dir_name(rec.name)
            count = used_dir_names.get(base_dir_name, 0)
            used_dir_names[base_dir_name] = count + 1
            # If duplicate display name appears, append wnid to avoid collisions.
            class_dir_name = base_dir_name if count == 0 else f"{base_dir_name}_{rec.wnid}"

            class_dir = images_root / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(1, args.samples_per_class + 1):
                out_img = class_dir / f"{sample_idx:04d}.png"

                # 为每个样本分配稳定且可复现的种子
                seed = args.base_seed + rec.index * 100000 + sample_idx
                scene_hint = choose_scene_hint(args.base_seed + sample_idx, rec.index)
                semantic_row = semantic_table.get(rec.wnid)
                if semantic_row:
                    prompt = build_disambiguated_prompt(
                        rec,
                        scene_hint,
                        semantic_row,
                        max_include_keywords=max(0, args.max_include_keywords),
                    )
                    negative_prompt = build_disambiguated_negative_prompt(
                        semantic_row,
                        max_exclude_keywords=max(0, args.max_exclude_keywords),
                    )
                else:
                    prompt = build_prompt(rec.name, scene_hint)
                    negative_prompt = NEGATIVE_PROMPT
                    if args.semantic_spec:
                        missing_semantics.append({
                            "wnid": rec.wnid,
                            "name": rec.name,
                        })

                row = {
                    "index": rec.index,
                    "wnid": rec.wnid,
                    "class_name": rec.name,
                    "class_dir_name": class_dir_name,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "semantic_sense": (semantic_row or {}).get("sense", None),
                    "semantic_include_keywords": (semantic_row or {}).get("include_keywords", []),
                    "semantic_exclude_keywords": (semantic_row or {}).get("exclude_keywords", []),
                    "seed": seed,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "samples_per_class": args.samples_per_class,
                    "size": "64x64",
                    "model_id": args.model_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "ok",
                    "output_path": str(out_img),
                }

                try:
                    if args.dry_run:
                        img = Image.new("RGB", (64, 64), (90, 90, 90))
                    else:
                        img = None
                        for attempt in range(args.max_retries + 1):
                            try_seed = seed + attempt * 7919
                            import torch

                            gen = torch.Generator(device=args.device).manual_seed(try_seed)
                            result = pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=args.steps,
                                guidance_scale=args.guidance_scale,
                                width=args.gen_size,
                                height=args.gen_size,
                                generator=gen,
                            )
                            img = result.images[0]
                            row["used_seed"] = try_seed
                            break
                        if img is None:
                            raise RuntimeError("generation failed after retries")

                    img = center_crop_resize_64(img)
                    img.save(out_img)
                    image_paths.append(out_img)

                except Exception as e:  # noqa: BLE001
                    row["status"] = "failed"
                    row["error"] = str(e)
                    failed_cases.append(
                        {
                            "index": rec.index,
                            "wnid": rec.wnid,
                            "class_name": rec.name,
                            "sample_idx": sample_idx,
                            "reason": str(e),
                            "prompt": prompt,
                        }
                    )

                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")

    make_preview_grid(image_paths, grid_path, cols=20)
    failed_cases_path.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    unique_missing = sorted({(x["wnid"], x["name"]) for x in missing_semantics})
    missing_rows = [{"wnid": w, "name": n} for w, n in unique_missing]
    missing_semantics_path.write_text(json.dumps(missing_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] grid: {grid_path}")
    print(f"[OK] failed_cases: {failed_cases_path} ({len(failed_cases)})")
    print(f"[OK] missing_semantics: {missing_semantics_path} ({len(missing_rows)})")


if __name__ == "__main__":
    main()

