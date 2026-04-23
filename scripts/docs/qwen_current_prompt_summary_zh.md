# 当前 Qwen 图像生成提示词总结

本文档整理的是当前脚本  
[generate_tiny_target_domain_qwen_api.py](/workspace/backdoor-toolbox-new1/scripts/generate_tiny_target_domain_qwen_api.py)  
里**实际生效**的提示词逻辑。

文档重点覆盖：

- 当前支持的两套提示词风格：`descriptive` 和 `catalog`
- 当前最常用的 `catalog` 模式的完整正/负提示词结构
- `catalog` 模式下固定 5 类型的具体内容
- 类别分组、短语义 cue、额外正向约束、额外负向约束
- 特殊 override 类的完整模板
- 影响提示词行为的关键命令行参数

本文档写作时的默认使用场景是：

```bash
python /workspace/backdoor-toolbox-new1/scripts/generate_tiny_target_domain_qwen_api.py \
  --class-map /workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json \
  --semantic-spec /workspace/data/tiny-target-domain-preview-1shot/tiny_class_semantics_v2_hardcases.json \
  --output-dir /workspace/data/tiny-target-domain-qwen-full \
  --model qwen-image-2.0 \
  --start-index 0 \
  --end-index 200 \
  --samples-per-class 5 \
  --size 1024x1024 \
  --prompt-style catalog \
  --catalog-include-keywords 1 \
  --disable-prompt-extend \
  --only-raw-and-64 \
  --raw-inside-images
```

在这类命令下，实际使用的是：

- `prompt-style = catalog`
- `catalog-include-keywords = 1`
- `samples-per-class = 5`
- `disable-prompt-extend = True`
- `disable-catalog-sample-variation = False`

所以你当前真正关心的核心，是 `catalog` 提示词系统。

---

## 1. 总览

当前脚本中存在两套提示词系统：

1. `descriptive`
2. `catalog`

入口参数在  
[generate_tiny_target_domain_qwen_api.py](/workspace/backdoor-toolbox-new1/scripts/generate_tiny_target_domain_qwen_api.py)
中的 `--prompt-style`。

默认值仍然是：

```text
descriptive
```

但你目前常用命令明确写了：

```text
--prompt-style catalog
```

因此，下面文档会把 `catalog` 作为主角来讲；`descriptive` 也会完整列出，但它现在不是你主流程的重点。

---

## 2. 当前 `descriptive` 模式

### 2.1 基础风格组件

当前 `descriptive` 模式使用这几个固定组件：

```text
STYLE_ANCHOR =
realistic color photograph, photorealistic DSLR look, natural materials and textures, accurate proportions, sharp focus

COMPOSITION_ANCHOR =
single main subject, class-defining details clearly visible, subject occupies most of the frame, simple uncluttered composition, no collage, no heavy occlusion

LIGHTING_ANCHOR =
balanced exposure, even lighting, natural color, no dramatic darkness
```

基础负提示词：

```text
illustration, cartoon, anime, painting, sketch, comic, digital art,
3d render, cgi, unreal engine, game asset, toy-like, plastic texture, waxy surface,
blurry, low resolution, low quality, jpeg artifacts, over-smoothed, noisy,
watermark, text, logo, border, frame, collage, grid, duplicate subject, multiple main subjects,
cropped subject, cut-off subject, tiny distant subject, severe occlusion, cluttered background,
underexposed, overexposed, harsh shadow, silhouette only, monochrome, grayscale
```

### 2.2 `descriptive` 基础模板

如果没有 semantic spec，当前模板是：

```text
A realistic color photo of {class_name}, {scene_hint},
single main subject, class-defining details clearly visible, subject occupies most of the frame,
simple uncluttered composition, no collage, no heavy occlusion,
balanced exposure, even lighting, natural color, no dramatic darkness,
realistic color photograph, photorealistic DSLR look, natural materials and textures, accurate proportions, sharp focus.
```

### 2.3 `descriptive + semantic spec` 模板

如果传了 `--semantic-spec`，并且该类在 semantic 表里存在条目，则模板变成：

```text
A realistic color photo of {class_name},
intended Tiny-ImageNet sense: {sense},
key visual cues: {include_keywords[:max_include_keywords]},
{scene_hint_override or scene_hint},
{rendered_framing_hint},
single main subject, class-defining details clearly visible, subject occupies most of the frame,
simple uncluttered composition, no collage, no heavy occlusion,
balanced exposure, even lighting, natural color, no dramatic darkness,
realistic color photograph, photorealistic DSLR look, natural materials and textures, accurate proportions, sharp focus.
```

### 2.4 `descriptive` 负提示词

结构是：

```text
NEGATIVE_PROMPT
+ semantic spec 中最多前 max_exclude_keywords 个 exclude_keywords
```

也就是：

```text
{基础 descriptive 负提示词}, {exclude_1}, {exclude_2}, {exclude_3}
```

### 2.5 备注

当前你的正式生成基本不走 `descriptive`。  
你现在主要在用 `catalog`。

---

## 3. 当前 `catalog` 模式总览

`catalog` 是你当前真正使用的系统。

它的核心目标不是“棚拍电商图”，而是：

- 统一在真实照片域
- 提供轻量语义钉住
- 每类在 `samples-per-class=5` 时按固定 5 类型生成
- 各类允许环境竞争注意力和一定的不完美抓拍感

### 3.1 当前统一风格后缀

当前 `catalog` 统一风格后缀是：

```text
class-defining subject or place remains recognizable, realistic everyday photo, natural color, non-studio composition
```

它的作用是：

- 保证仍然是写实照片
- 保证类别语义仍然可识别
- 明确不是棚拍或干净产品图

---

## 4. 当前 `catalog` 正提示词模板

### 4.1 普通类模板

当前普通类的 `catalog` 正提示词结构是：

```text
A clean reference natural photograph of {class_name},
[showing {catalog_cue}],
[scene/place/common extra guidance],
[fixed scenario profile],
class-defining subject or place remains recognizable, realistic everyday photo, natural color, non-studio composition.
```

更精确地说，拼接顺序是：

1. `A clean reference natural photograph of {class_name}`
2. 如果有 `cue`，加 `showing {cue}`
3. 如果是 scene 类，追加 scene addon
4. 如果是 place 类，追加 place addon
5. 如果该类命中了 `EXTRA_CATALOG_GUIDANCE_BY_WNID`，追加类专属正向约束
6. 如果启用了 sample variation，追加固定五类型之一
7. 最后加统一风格后缀

### 4.2 override 类模板

如果类别在 `CLASS_SPECIFIC_CATALOG_PROMPT_OVERRIDES` 中：

- 先使用该类的完整 override prompt
- 再在结尾追加固定五类型之一

结构是：

```text
{class_specific_override_prompt}, {fixed scenario profile}.
```

---

## 5. 当前 `catalog` 固定五类型

当前并不是从多个短语池随机抽词来拼接。  
实际逻辑是：

- 当 `samples-per-class > 1`
- 且没有加 `--disable-catalog-sample-variation`
- `sample_idx = 1..5` 会固定映射到 5 个槽位

代码逻辑等价于：

```text
sample 1 -> 固定类型 1
sample 2 -> 固定类型 2
sample 3 -> 固定类型 3
sample 4 -> 固定类型 4
sample 5 -> 固定类型 5
```

也就是说，当前的“采样变化”不是随机，而是固定 5 种摄影原型。

---

## 6. 当前三类分组

当前 sample variation 只分成三组：

1. `object`
2. `animal`
3. `scene`

当前逻辑是：

- 若 `wnid` 在 `ANIMAL_VARIANT_WNIDS`，走 `animal`
- 否则若 `wnid` 在 `SCENE_LIKE_WNIDS`，走 `scene`
- 否则走 `object`

所以严格说：

- 不是所有类都一套 prompt
- 也不是“四五十个细分组”
- 当前实际生效的是三大组 + 少量 override 类

---

## 7. 当前 `object` 五类型

### `sample 1`

```text
closer canonical real-world view, object relatively large in frame, frontal or near-frontal eye-level angle, off-center, everyday setting, one nearby object, surface edge, or foreground element occupies noticeable area and competes strongly for attention while key class-defining parts remain readable
```

人话解释：

- 近景
- 接近正面
- 物体较大
- 不居中
- 有明显前景或旁边物体抢注意力

### `sample 2`

```text
high-angle wider context view, clearly elevated or downward-looking viewpoint, object medium size in frame, surrounding room, workspace, shop, street, or storage context visible, several nearby objects or surfaces take substantial space and the object need not be the most salient region
```

人话解释：

- 明显高视角/俯视
- 更宽环境
- 物体中等大小
- 周围环境占较大面积
- 主体不一定最显眼

### `sample 3`

```text
strong lateral side-view, clear side-profile or oblique side angle, object close to one frame edge, non-critical parts may approach the border, a nearby object, shelf edge, wall edge, or table edge occupies noticeable space and competes strongly for attention
```

人话解释：

- 强侧拍
- 主体贴近边缘
- 非关键部位可接近边框
- 旁边结构或物体明显抢注意力

### `sample 4`

```text
low-angle cluttered real-world view, clearly lower or upward-looking viewpoint, stronger depth perspective, multiple nearby objects, furniture, tools, packaging, or structural elements take substantial space, the object remains readable but competes with several strong visual elements
```

人话解释：

- 明显低机位/仰视感
- 景深和纵深更强
- 环境忙乱
- 多个强视觉元素同时存在

### `sample 5`

```text
imperfect off-axis snapshot, strong passing or turning viewpoint, partial foreground blockage, edge cutoff, uneven exposure, or focus falloff allowed, a foreground element or nearby object may take large visible area and rival the labeled object for attention while overall identity remains clear
```

人话解释：

- 强 off-axis 抓拍
- 路过式、转向式视角
- 允许前景遮挡、边缘切入、曝光波动、轻微软焦
- 干扰最重

---

## 8. 当前 `animal` 五类型

### `sample 1`

```text
closer canonical animal view, animal relatively large in frame, frontal or near-frontal eye-level viewpoint, off-center, habitat visible, key species traits readable, foreground texture, vegetation, or another habitat element occupies noticeable area and competes strongly for attention
```

人话解释：

- 近景
- 近正面
- eye-level
- 动物较大
- 但前景纹理、植被或 habitat 元素占明显面积

### `sample 2`

```text
high-angle wider habitat view, clearly elevated or downward-looking observational viewpoint, animal medium size in frame, broader environment visible, surrounding habitat elements take substantial space and the animal is not necessarily the strongest visual patch
```

人话解释：

- 明显高视角/向下看
- 更宽 habitat
- 动物只占中等比例
- 环境占较大面积
- 动物不一定是最显眼区域

### `sample 3`

```text
strong side-profile sighting, clear lateral side view rather than a frontal portrait, animal close to one frame edge, non-critical body parts may approach the border, side vegetation, rocks, water edges, or nearby habitat structure occupies noticeable space and competes strongly for attention
```

人话解释：

- 明显侧拍
- 侧身轮廓
- 贴边
- 侧边植被/岩石/水边/结构占明显面积

### `sample 4`

```text
low-angle busy habitat view, clearly lower or upward-looking ground-near viewpoint, stronger foreground-background layering, grass, branches, stones, water, or habitat clutter takes substantial space, the animal remains readable but competes with a visually busy environment
```

人话解释：

- 明显低机位
- 靠近地面
- 前后景层次更强
- 草、树枝、石头、水面等占大块面积
- 环境明显更忙

### `sample 5`

```text
imperfect off-axis animal snapshot, strong passing, turning, or oblique sighting viewpoint, partial screening, foreground obstruction, focus softness, or uneven exposure allowed, nearby habitat elements may take large visible area and rival the animal for attention while species identity remains clear
```

人话解释：

- 强 off-axis 动物抓拍
- 路过/转向/斜角目击
- 允许前景遮挡、曝光波动、轻微软焦
- 允许 habitat 元素大面积入画并与动物抢注意力

---

## 9. 当前 `scene` 五类型

### `sample 1`

```text
closer canonical scene view, ordinary eye-level real-location viewpoint, main place or structure cue clearly readable, off-center, foreground and side elements occupy noticeable space and compete strongly for attention
```

人话解释：

- 普通 eye-level 现场图
- 主语义区偏心
- 前景和侧边元素占明显面积

### `sample 2`

```text
high-angle wider establishing view, clearly elevated or overlook-like viewpoint from farther away, the key place cue occupies only part of the layout, broader surroundings participate in recognition, secondary scene elements take substantial space and the main cue need not dominate the frame
```

人话解释：

- 更高、更远的 establishing view
- 主语义区只占布局一部分
- 周围环境大面积参与识别

### `sample 3`

```text
strong lateral or roadside scene view, diagonal side-view rather than a straight-on composition, main semantic region close to one side of the frame, side walls, rails, rocks, vegetation, or boundary structures occupy noticeable space and compete strongly for attention
```

人话解释：

- 强侧视/斜视
- 主语义区贴近一侧
- 栏杆、岩石、植被、边界结构等占明显面积

### `sample 4`

```text
low-angle busy real-world scene, clearly lower or upward-emphasizing viewpoint, stronger depth perspective, layered foreground and background, rails, rocks, vegetation, signs, walls, or vehicles take substantial space while place identity stays clear, the most salient patch need not be the class cue
```

人话解释：

- 明显低机位/仰视感
- 更强纵深
- 真实场景更忙、更杂
- 最显眼区域不一定就是类别语义区

### `sample 5`

```text
imperfect off-axis scene snapshot, strong passing viewpoint, uneven exposure, haze, reflections, weathering, foreground edge clutter, or partial occluders allowed, non-central elements may take large visible area and rival the main place cue while the scene remains semantically clear
```

人话解释：

- 强 off-axis 场景抓拍
- 路过视角
- 允许反光、雾感、边缘杂乱、遮挡、曝光问题
- 允许非中心元素占大面积并抢注意力

---

## 10. 当前 `catalog` cue 选择逻辑

当前 `catalog` 模式里的 `cue` 选择顺序是：

1. 先查 `MINIMAL_CATALOG_CUES`
2. 如果没有，再从 semantic spec 的 `include_keywords` 里取
3. 但会过滤掉太弱、太冗余的 cue

当前过滤逻辑包括：

- 如果 keyword 和类名完全一样，跳过
- 如果 keyword 只是类名的一部分，也跳过
  - 例如 `Egyptian cat` 的 `egyptian`
- 如果 keyword 的 token 集合只是类名 token 的子集，也跳过

这一步是为了避免出现低价值 cue，比如：

```text
showing egyptian
```

---

## 11. 当前 `MINIMAL_CATALOG_CUES`

这部分是最重要的“短语义钉住”。

当前完整内容如下：

- `apron` -> `full apron garment clearly visible`
- `alp` -> `high mountain slope landscape`
- `altar` -> `religious altar without people`
- `bannister` -> `stair handrail and balusters`
- `barbershop` -> `barbershop storefront or empty interior without people`
- `bathtub` -> `full-size bathtub clearly visible`
- `beacon` -> `lighthouse beacon tower`
- `brass` -> `plaque`
- `brain coral` -> `brain coral colony`
- `broom` -> `natural straw broom`
- `butcher shop` -> `butcher shop storefront or empty meat market without people`
- `CD player` -> `full electronic CD player device with buttons and disc tray clearly visible, not a loose compact disc`
- `Chihuahua` -> `very small Chihuahua dog with large upright ears`
- `computer keyboard` -> `standalone full computer keyboard occupying most of the frame on a simple desk surface`
- `desk` -> `desk furniture as the main subject`
- `dining table` -> `full dining table furniture, not a meal scene`
- `organ` -> `pipe organ`
- `crane` -> `construction crane`
- `cliff` -> `steep rocky cliff face`
- `cliff dwelling` -> `ancient stone dwelling built into a cliff`
- `comic book` -> `closed comic book as the main object`
- `confectionery` -> `candy shop storefront or display without people`
- `reel` -> `spool`
- `dam` -> `large concrete dam structure`
- `walking stick` -> `stick insect`
- `guacamole` -> `bowl of guacamole dip`
- `lakeside` -> `empty inland freshwater lakeshore with calm still water, no people, no ocean, no beach, no surf`
- `nail` -> `single large metal nail fastener as the main subject`
- `torch` -> `flaming torch or fire torch, not a flashlight`
- `parking meter` -> `single parking meter device`
- `seashore` -> `coastline shoreline landscape without people`
- `pole` -> `utility pole`
- `Labrador retriever` -> `short-haired Labrador retriever dog`
- `golden retriever` -> `long-haired golden retriever dog with feathered golden coat`
- `chest` -> `full dresser furniture with drawers clearly visible`
- `poncho` -> `full rain poncho waterproof outer garment clearly visible`
- `sports car` -> `full sports car clearly visible`
- `sunglasses` -> `full pair of sunglasses clearly visible`
- `thatch` -> `broad full thatched roof section clearly visible`
- `vestment` -> `full liturgical vestment robe clearly visible as the main subject`
- `plate` -> `round dinner plate`
- `projectile` -> `metal projectile`
- `flagpole` -> `upright flagpole`
- `viaduct` -> `arched stone or concrete viaduct bridge`

---

## 12. 当前场景/地点统一附加语句

### 12.1 scene 类统一附加

```text
main place, structure, or landscape clearly recognizable, no dominant foreground person
```

### 12.2 place 类统一附加

```text
empty or person-light place view, storefront or interior clearly recognizable
```

---

## 13. 当前额外正向约束 `EXTRA_CATALOG_GUIDANCE_BY_WNID`

这部分会继续追加到正提示词中。

- `bathtub` -> `full object visible, not cropped`
- `CD player` -> `audio device itself is the dominant main subject, buttons and disc tray clearly visible, full device visible, not cropped, no loose disc as the main subject`
- `chest` -> `full furniture visible, not cropped`
- `computer keyboard` -> `keyboard alone dominates the image, keyboard fills most of the frame, full keyboard clearly visible, not cropped, on a simple desk surface, no person, no face, no hands, no headset, no monitor, no laptop, no other main objects`
- `desk` -> `desk itself is the main subject, any person must be secondary and not dominant`
- `dining table` -> `table furniture itself is the main subject, no dominant meal`
- `nail` -> `no hand holding it, no person`
- `organ` -> `musical instrument clearly recognizable, not anatomy`
- `poncho` -> `poncho itself is the main subject, full poncho silhouette visible, not cropped, clearly a rain poncho, not a regular shirt or coat`
- `sports car` -> `full vehicle visible, not cropped`
- `sunglasses` -> `full object visible, not cropped`
- `thatch` -> `large continuous thatched roof area clearly recognizable, not just a small edge or fragment`
- `vestment` -> `vestment itself is the main subject, full ceremonial robe visible, a person may wear it but the garment must dominate, not a portrait crop`
- `lakeside` -> `inland freshwater lakeshore itself is the main subject, empty scene, no people, calm still water, inland setting, no surf, no ocean horizon`
- `apron` -> `apron itself is the dominant main subject, full garment visible, not cropped, simple kitchen or cooking setting`

---

## 14. 当前 override 类完整模板

以下类不走普通模板，而是走完整 override prompt：

### 14.1 `computer keyboard`

```text
A clean reference natural photograph of computer keyboard, showing a standalone full computer keyboard on a simple desk surface, keyboard occupies most of the frame and is the only main subject, full keyboard clearly visible, not cropped, no person, no face, no hands, no monitor, no laptop, no other main objects, clear main subject, realistic real-world setting, clean composition, sharp focus, natural color, visually consistent reference-photo style.
```

### 14.2 `poncho`

```text
A realistic candid outdoor photograph of poncho, showing a person naturally wearing a full waterproof rain poncho in light rain, the poncho garment is the main subject, full poncho silhouette clearly visible, realistic human posture and proportions, natural folds and wet fabric texture, person is secondary, face visible but not emphasized, clearly a practical rain poncho and not a regular coat or shirt, full garment visible and not cropped, real outdoor rainy setting, clean composition, sharp focus, natural color, visually consistent reference-photo style.
```

### 14.3 `thatch`

```text
A clean reference natural photograph of thatch, showing a traditional hut or house with a full thatched roof clearly visible, the thatched roof is the main subject, most of the roof visible in frame, roof material and overall roof form clearly recognizable, clear main subject, realistic real-world setting, clean composition, sharp focus, natural color, visually consistent reference-photo style.
```

### 14.4 `lakeside`

```text
A clean reference natural photograph of lakeside, showing an inland freshwater lakeshore with calm still lake water, grassy or rocky shoreline, and distant treeline on the opposite shore, empty scene with no people, no ocean, no sea, no beach, no surf, clear main subject, realistic real-world setting, clean composition, sharp focus, natural color, visually consistent reference-photo style.
```

### 14.5 override 类与五类型的关系

override 类不是完全固定死。

当前行为是：

- 先用 override 基础 prompt
- 再追加该类所属组的固定五类型之一

例如：

- `computer keyboard` 用 override + `object` 五类型
- `lakeside` 用 override + `scene` 五类型

---

## 15. 当前 `catalog` 负提示词

### 15.1 基础负提示词

```text
illustration, cartoon, anime, painting, sketch, 3d render, cgi, blurry, low quality, noisy,
jpeg artifacts, watermark, text, logo, multiple main subjects, duplicate subject, tiny distant subject,
severe occlusion, underexposed, overexposed, harsh shadow, readable text,
dramatic perspective, extreme camera angle, stylized lighting, oversaturated color
```

### 15.2 scene 类统一追加

```text
portrait, selfie, close-up face, dominant foreground person, crowd
```

### 15.3 place 类统一追加

```text
customer, customers, barber cutting hair, butcher handling meat, cashier, shopkeeper, person behind counter
```

### 15.4 当前额外负约束 `EXTRA_CATALOG_NEGATIVE_BY_WNID`

- `bathtub` -> `sink, washbasin, basin`
- `broom` -> `green plastic broom, neon green broom`
- `CD player` -> `loose compact disc only, single compact disc, cd disc only, jewel case, disc in hand, disc as the main subject`
- `chest` -> `cropped furniture, partial furniture`
- `computer keyboard` -> `cropped keyboard, partial keyboard, laptop only, person, face, portrait, headset, headphones, hands on keyboard, person using keyboard, portrait with keyboard, monitor as main subject, desk scene with person, keyboard as a secondary object, office scene as the main subject, computer setup as the main subject, mouse as the main subject`
- `desk` -> `person as the sole subject, office worker portrait, dominant foreground person`
- `dining table` -> `meal scene, dinner plate as main subject, people dining`
- `nail` -> `hand holding nail, person holding object`
- `organ` -> `human organ, anatomy, surgery, biology diagram`
- `poncho` -> `blanket, shawl, regular shirt, coat, jacket, cropped garment, mannequin, hollow garment, floating garment, faceless void, fashion editorial, costume cape, glossy plastic costume`
- `sports car` -> `cropped car, partial vehicle`
- `sunglasses` -> `person wearing sunglasses, cropped eyewear`
- `thatch` -> `cropped roof, partial roof edge only, tiny roof fragment`
- `vestment` -> `cropped robe, portrait, face as the main subject, person wearing robe as portrait`
- `torch` -> `flashlight, electric torch`
- `lakeside` -> `person, people, human, figure in scene, person standing by the lake, beach scene, ocean coast, sea, seashore, coastline, surf, waves, tropical beach, sandy ocean shore, ocean horizon, palm trees, beach umbrellas`
- `apron` -> `fashion portrait, model wearing apron as the main subject, plain white studio backdrop`

### 15.5 当前 `catalog` 负提示词总结构

```text
CATALOG_NEGATIVE_PROMPT
+ [SCENE_CATALOG_NEGATIVE_PROMPT_SUFFIX if scene-like]
+ [PLACE_CATALOG_NEGATIVE_PROMPT_SUFFIX if place-like]
+ [EXTRA_CATALOG_NEGATIVE_BY_WNID if exists]
```

---

## 16. 当前分组集合

### 16.1 `SCENE_LIKE_WNIDS`

当前 scene-like 类包括：

- alp
- altar
- bannister
- barbershop
- beacon
- brain coral
- butcher shop
- cliff
- cliff dwelling
- confectionery
- dam
- lakeside
- seashore
- viaduct

### 16.2 `PLACE_WNIDS`

当前 place 类包括：

- altar
- barbershop
- butcher shop
- confectionery

### 16.3 `ANIMAL_VARIANT_WNIDS`

当前动物组已扩展到大量真实动物类，包含：

- 金鱼、火蝾螈、牛蛙、鳄鱼、蟒蛇、蝎子、蜘蛛、蜈蚣
- 鹅、企鹅、信天翁
- Chihuahua、Yorkshire terrier、golden retriever、Labrador retriever、German shepherd、poodle
- tabby、Persian cat、Egyptian cat、cougar、lion、brown bear
- ladybug、fly、bee、grasshopper、cockroach、mantis、dragonfly、butterflies
- guinea pig、hog、ox、bison、bighorn、gazelle、camel
- orangutan、chimpanzee、baboon、African elephant、lesser panda
- jellyfish、dugong
- trilobite

这些类现在都走 `animal` 的固定五类型。

---

## 17. 旧的 `*_VARIANTS` 列表说明

脚本里仍然保留了以下列表：

- `OBJECT_CATALOG_VIEW_VARIANTS`
- `OBJECT_CATALOG_CONTEXT_VARIANTS`
- `OBJECT_CATALOG_CAPTURE_VARIANTS`
- `OBJECT_CATALOG_STATE_VARIANTS`
- `OBJECT_CATALOG_DISTRACTOR_VARIANTS`
- `ANIMAL_*_VARIANTS`
- `SCENE_*_VARIANTS`

这些列表本身仍然定义在代码里，但**当前固定五类型逻辑并不直接使用它们来拼 prompt**。

当前真正用于 sample variation 的，是：

- `OBJECT_CATALOG_SCENARIO_PROFILES`
- `ANIMAL_CATALOG_SCENARIO_PROFILES`
- `SCENE_CATALOG_SCENARIO_PROFILES`

也就是说：

- 这些细粒度 `VARIANTS` 列表目前更像历史遗留或备用素材
- 当前真正生效的是“三组固定五类型”

---

## 18. 当前影响提示词行为的关键参数

### `--prompt-style`

- `descriptive`
- `catalog`

你当前主要用的是：

```text
catalog
```

### `--catalog-include-keywords`

当前你常用：

```text
1
```

作用：

- 最多从 semantic spec 中取 1 个 include keyword 来做 cue
- 但若命中 `MINIMAL_CATALOG_CUES`，优先用手工 cue

### `--disable-catalog-sample-variation`

如果开启这个参数：

- 固定五类型会被关闭
- 每类多张图的 prompt 差异会显著减少

当前你通常**不会开**它。

### `--samples-per-class`

如果：

```text
samples-per-class = 1
```

则：

- 不追加 sample variation

如果：

```text
samples-per-class = 5
```

则：

- `sample 1..5` 分别使用固定五类型

### `--disable-prompt-extend`

当前你常用这个参数。

作用：

- 关闭 Qwen 服务端的 prompt 扩写
- 让本地脚本拼出来的 prompt 更接近实际发送内容

### `--negative-prompt`

如果你手动传这个参数：

- 会直接覆盖脚本自动生成的负提示词

当前你大多数命令**没有显式传它**，所以会使用脚本内部自动构造的 negative prompt。

---

## 19. 当前最常见实际情况

在你当前最常用的命令下：

```bash
--prompt-style catalog
--catalog-include-keywords 1
--samples-per-class 5
--disable-prompt-extend
```

真正生效的东西是：

- `catalog` 主模板
- 轻量 cue
- 固定五类型
- 自动 negative prompt
- 不经过服务端 prompt 扩写

所以，你现在每个类别的 5 张图，本质上是：

1. 固定类名和可能的 cue
2. 固定类专属正向约束
3. 固定五类型之一
4. 固定统一风格后缀
5. 固定自动负提示词

---

## 20. 一句话总结

当前这版提示词系统的核心不是“随机拼很多短语”，而是：

- 以 `catalog` 为主
- 用短 cue 防止语义跑偏
- 用三大类 `object / animal / scene`
- 用固定五类型保证同类内部差异
- 用额外正/负约束处理高风险类别
- 用统一风格后缀保持整个数据集处于同一个真实照片域

如果你后面再改 prompt，最值得优先关注的部分是：

1. `OBJECT/ANIMAL/SCENE_CATALOG_SCENARIO_PROFILES`
2. `MINIMAL_CATALOG_CUES`
3. `EXTRA_CATALOG_GUIDANCE_BY_WNID`
4. `EXTRA_CATALOG_NEGATIVE_BY_WNID`
5. `CLASS_SPECIFIC_CATALOG_PROMPT_OVERRIDES`
