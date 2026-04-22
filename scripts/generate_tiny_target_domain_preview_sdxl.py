#!/usr/bin/env python3
"""与 generate_tiny_target_domain_preview_flux.py 相同的数据布局与 manifest，模型改为 Stable Diffusion XL Base 1.0。

用法（在项目根目录）：
  python scripts/generate_tiny_target_domain_preview_sdxl.py --output-dir ./data/my-sdxl-run ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 复用 FLUX 脚本中的提示词、class_map、语义表、裁剪与拼图
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_flux_preview",
    _SCRIPTS / "generate_tiny_target_domain_preview_flux.py",
)
_flux = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_flux)

apply_hf_endpoint = _flux.apply_hf_endpoint
build_disambiguated_negative_prompt = _flux.build_disambiguated_negative_prompt
build_disambiguated_prompt = _flux.build_disambiguated_prompt
build_prompt = _flux.build_prompt
center_crop_resize_64 = _flux.center_crop_resize_64
choose_scene_hint = _flux.choose_scene_hint
load_class_map = _flux.load_class_map
load_semantic_spec = _flux.load_semantic_spec
make_preview_grid = _flux.make_preview_grid
sanitize_class_dir_name = _flux.sanitize_class_dir_name
NEGATIVE_PROMPT = _flux.NEGATIVE_PROMPT


def get_sdxl_pipeline(model_id: str, device: str, dtype: str, offload: str, local_files_only: bool = False):
    import torch
    from diffusers import StableDiffusionXLPipeline

    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    kwargs: Dict = {"torch_dtype": torch_dtype, "use_safetensors": True, "local_files_only": local_files_only}
    if dtype == "float16":
        kwargs["variant"] = "fp16"

    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)

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
        except Exception as e:  # noqa: BLE001
            print(f"[Memory] enable_sequential_cpu_offload failed: {e}; fallback")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=False)
    return pipe


def align_gen_size(n: int) -> int:
    """SDXL 要求宽高为 8 的倍数。"""
    return max(256, (n // 8) * 8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tiny target-domain set with SDXL Base 1.0.")
    parser.add_argument(
        "--class-map",
        type=str,
        default="./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
    )
    parser.add_argument("--output-dir", type=str, default="./data/tiny-target-domain-sdxl-preview")
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Hugging Face 上的 SDXL Base 模型 ID",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="SDXL 常用 float16（fp16 权重）",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--gen-size",
        type=int,
        default=1024,
        help="生成边长（会向下对齐到 8 的倍数），再中心裁剪为 64x64 保存",
    )
    parser.add_argument("--base-seed", type=int, default=20260414)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument(
        "--offload",
        type=str,
        choices=["auto", "none", "model", "sequential"],
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-hf-mirror", action="store_true")
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--semantic-spec", type=str, default=None)
    parser.add_argument("--strict-disambiguation", action="store_true")
    parser.add_argument("--max-include-keywords", type=int, default=3)
    parser.add_argument("--max-exclude-keywords", type=int, default=3)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅使用本地 HF 缓存（离线）",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        apply_hf_endpoint(args.hf_endpoint)
    elif args.use_hf_mirror:
        apply_hf_endpoint("https://hf-mirror.com")

    class_map_path = Path(args.class_map).resolve()
    out_root = Path(args.output_dir).resolve()
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    classes = load_class_map(class_map_path)
    semantic_spec_path = Path(args.semantic_spec).resolve() if args.semantic_spec else None
    semantic_table = load_semantic_spec(semantic_spec_path)
    missing_semantics: List[Dict] = []

    pipe = None
    if not args.dry_run:
        pipe = get_sdxl_pipeline(
            args.model_id, args.device, args.dtype, args.offload, local_files_only=args.local_files_only
        )

    manifest_path = out_root / "manifest_preview_1shot.jsonl"
    failed_cases_path = out_root / "failed_cases.json"
    missing_semantics_path = out_root / "missing_semantics.json"
    grid_path = out_root / "preview_grid.png"

    image_paths: List[Path] = []
    failed_cases: List[Dict] = []
    used_dir_names: Dict[str, int] = {}

    gen_w = align_gen_size(args.gen_size)
    gen_h = gen_w

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rec in classes:
            base_dir_name = sanitize_class_dir_name(rec.name)
            count = used_dir_names.get(base_dir_name, 0)
            used_dir_names[base_dir_name] = count + 1
            class_dir_name = base_dir_name if count == 0 else f"{base_dir_name}_{rec.wnid}"

            class_dir = images_root / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx in range(1, args.samples_per_class + 1):
                out_img = class_dir / f"{sample_idx:04d}.png"
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
                        missing_semantics.append({"wnid": rec.wnid, "name": rec.name})

                row: Dict = {
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
                    "backend": "sdxl",
                    "gen_width": gen_w,
                    "gen_height": gen_h,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "ok",
                    "output_path": str(out_img),
                }

                try:
                    if args.dry_run:
                        from PIL import Image

                        img = Image.new("RGB", (64, 64), (90, 90, 90))
                    else:
                        import torch

                        img = None
                        for attempt in range(args.max_retries + 1):
                            try_seed = seed + attempt * 7919
                            gen = torch.Generator(device=args.device).manual_seed(try_seed)
                            result = pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=args.steps,
                                guidance_scale=args.guidance_scale,
                                width=gen_w,
                                height=gen_h,
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


if __name__ == "__main__":
    main()
