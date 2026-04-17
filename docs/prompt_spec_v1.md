# Tiny Target Domain Prompt Spec V1

## Purpose
用于 `FLUX.1 schnell` 的 Tiny-ImageNet 目标域预览生成（200 类，每类 1 张，64×64）。

## Global Style Anchor
- 真实摄影：自然光、真实材质、真实纹理
- 统一视觉：色调一致、纪实风格、轻微景深
- 非艺术风：禁止插画、卡通、3D 渲染感

## Positive Prompt Template
`A realistic natural photograph of {class_name}, {scene_hint}. natural documentary photography, realistic camera capture, coherent color science, soft natural lighting, subtle depth of field, real-world textures, no studio fantasy look, square composition, 64x64.`

## Negative Prompt
`illustration, cartoon, anime, painting, sketch, comic, 3d render, cgi, plastic texture, low quality, blurry, watermark, text, logo`

## Scene Hint Pool
- `centered subject, clean composition`
- `outdoor context with natural perspective`
- `lifelike details with believable background`
- `documentary framing with realistic context`

## Reproducibility
- 每类固定种子：`seed = base_seed + class_index`
- 失败重试时仅改 seed，不改 prompt
- manifest 必填：`wnid/class_name/prompt/negative_prompt/seed/steps/guidance_scale/model_id/output_path/status`

