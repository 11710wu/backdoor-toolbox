#!/usr/bin/env python3
"""Generate Tiny target-domain images with Qwen Image API.

特点：
- 默认模型为 `qwen-image-2.0`
- 当模型为 `qwen-image-*` 时，自动使用官方 DashScope 文生图接口
- 沿用当前仓库的 Tiny-ImageNet 类映射与 prompt 构造逻辑
- 默认每个类别生成 5 张
- 默认生成更高分辨率的 `2048x2048` 原图
- 同时保存原始高分辨率图 + 128x128 调试图 + 64x64 评测图 + 64x64 轻微锐化图
- 写出 manifest，便于后续复查、筛选和组织数据集

示例：
  export DASHSCOPE_API_KEY=sk-xxx
  python scripts/generate_tiny_target_domain_qwen_api.py \
    --class-map ./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json \
    --output-dir ./data/tiny-target-domain-qwen-5shot \
    --samples-per-class 5 \
    --semantic-spec ./data/tiny-target-domain-preview-1shot/semantic_spec.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps

from generate_tiny_target_domain_preview_flux import (
    choose_scene_hint,
    load_class_map,
    load_semantic_spec,
    sanitize_class_dir_name,
)


STYLE_ANCHOR = (
    "realistic color photograph, photorealistic DSLR look, natural materials and textures, "
    "accurate proportions, sharp focus"
)


COMPOSITION_ANCHOR = (
    "single main subject, class-defining details clearly visible, subject occupies most of the frame, "
    "simple uncluttered composition, no collage, no heavy occlusion"
)

LIGHTING_ANCHOR = (
    "balanced exposure, even lighting, natural color, no dramatic darkness"
)

NEGATIVE_PROMPT = (
    "illustration, cartoon, anime, painting, sketch, comic, digital art, "
    "3d render, cgi, unreal engine, game asset, toy-like, plastic texture, waxy surface, "
    "blurry, low resolution, low quality, jpeg artifacts, over-smoothed, noisy, "
    "watermark, text, logo, border, frame, collage, grid, duplicate subject, multiple main subjects, "
    "cropped subject, cut-off subject, tiny distant subject, severe occlusion, cluttered background, "
    "underexposed, overexposed, harsh shadow, silhouette only, monochrome, grayscale"
)

REFERENCE_PHOTO_STYLE_PROMPT_SUFFIX = (
    "class-defining subject or place remains recognizable, realistic everyday photo, natural color, non-studio composition"
)

SCENE_REFERENCE_PROMPT_ADDON = (
    "main place, structure, or landscape clearly recognizable, no dominant foreground person"
)

PLACE_REFERENCE_PROMPT_ADDON = (
    "empty or person-light place view, storefront or interior clearly recognizable"
)

OBJECT_CATALOG_VIEW_VARIANTS = [
    "slightly different viewpoint",
    "slightly elevated viewpoint",
    "slightly low-angle viewpoint",
    "mild three-quarter view",
    "slightly different subject distance",
    "more frontal eye-level view",
    "more side-biased view",
    "wider normal-lens view",
    "noticeably wider mid-distance view",
    "more oblique off-axis everyday viewpoint",
    "subject seen from a passing side-angle rather than a deliberate showcase view",
]

OBJECT_CATALOG_CONTEXT_VARIANTS = [
    "slightly tighter framing with key details clearly visible",
    "slightly wider framing with simple surrounding context",
    "main subject slightly off-center but still dominant",
    "moderate real-world background context without clutter",
    "natural object placement with clear separation from the background",
    "subject placed near one side of the frame with balanced empty space",
    "broader surrounding context visible while the subject remains dominant",
    "more environmental context visible without distracting from the subject",
    "subject occupies only a moderate portion of the frame with meaningful nearby context",
    "subject seen within a busier room, shelf, workbench, or street setting while key cues remain visible",
    "subject positioned near an image edge with non-critical parts close to the border while key cues remain visible",
    "subject read from a lived-in environment rather than a clean isolated arrangement",
]

OBJECT_CATALOG_CAPTURE_VARIANTS = [
    "soft natural lighting",
    "gentle side lighting with clear focus",
    "ordinary everyday snapshot feel while keeping the main subject clear",
    "subtle depth variation between subject and background",
    "slight handheld photo feel but still sharp",
    "mild indoor ambient lighting with realistic color cast",
    "overcast outdoor-like soft light with reduced contrast",
    "clear flash-free phone-camera snapshot feel",
    "slightly uneven phone-camera autoexposure while class-defining parts remain clear",
    "mixed practical lighting from the surrounding environment with realistic tonal variation",
    "slight focus falloff or consumer-camera imperfection away from the key class cue",
    "mild reflection, glare, or surface shine may be present in a realistic way",
]

OBJECT_CATALOG_STATE_VARIANTS = [
    "subject shown with minimal surrounding distraction",
    "subject shown in a natural real-world placement",
    "subject shown with mild surrounding context but still as the only dominant subject",
    "subject shown from a slightly farther ordinary viewing distance",
    "subject shown in a more candid non-studio arrangement",
    "subject shown as one important element within a busier everyday setting",
    "subject shown with recognizability preserved but not necessarily filling the frame",
    "subject appearing in ordinary use or in situ rather than carefully displayed",
    "subject seen from a more incidental pass-by viewpoint rather than a deliberate showcase angle",
    "subject encountered within an active everyday setting rather than arranged for the camera",
    "subject shown in a believable lived-in scene with imperfect framing",
]

OBJECT_CATALOG_DISTRACTOR_VARIANTS = [
    "secondary everyday objects may be visible nearby without changing the class identity",
    "mild visual competition from nearby context objects is allowed while key class-defining parts stay visible",
    "a minor foreground overlap or edge distraction is allowed if the subject remains identifiable",
    "the subject need not be the only attention-grabbing element in the frame",
    "broader room, street, or tabletop context may compete mildly for attention",
    "small non-target details can share the frame while the labeled object remains recognizable",
    "nearby furniture, tools, packaging, or surfaces may share attention in a believable everyday way",
    "foreground edge elements, reflections, or surrounding clutter may mildly interfere without hiding the main class cue",
    "one nearby object or surface can attract comparable attention while the labeled object remains readable",
    "a partial foreground blocker or frame-edge cutoff may affect non-critical parts in a realistic snapshot way",
]

ANIMAL_CATALOG_VIEW_VARIANTS = [
    "slightly different viewpoint",
    "slightly different pose",
    "head turned slightly",
    "mild three-quarter animal view",
    "slightly different subject distance",
    "more side-on animal view",
    "more frontal animal view",
    "slightly lower eye-level animal view",
    "more distant field-observation view",
    "casual off-axis sighting angle",
    "animal noticed from a less direct sideward viewpoint",
]

ANIMAL_CATALOG_CONTEXT_VARIANTS = [
    "slightly tighter framing with the animal still fully recognizable",
    "slightly wider framing with habitat context visible",
    "main subject slightly off-center but still dominant",
    "natural foreground and background layering",
    "realistic environment context without distracting clutter",
    "animal placed nearer one edge of the frame with open surrounding space",
    "more habitat detail visible around the animal",
    "key species-defining body parts remain clear against natural context",
    "animal occupies only a moderate portion of the frame within a wider habitat view",
    "animal seen through a more layered natural environment while species cues remain readable",
    "animal positioned near the frame edge while the key species cue remains visible",
    "animal read from a busier habitat layout with multiple natural depth layers",
]

ANIMAL_CATALOG_CAPTURE_VARIANTS = [
    "soft natural lighting",
    "slight motion cue while the animal remains clearly visible",
    "ordinary wildlife snapshot feel with clear focus",
    "natural shadow variation",
    "subtle depth variation between animal and background",
    "soft cloudy daylight with gentle contrast",
    "ordinary phone-camera animal snapshot feel",
    "mild motion energy while species-defining features stay clear",
    "slightly imperfect casual sighting capture with realistic tonal variation",
    "field-snapshot rendering with everyday camera limitations but recognizable species traits",
    "slight motion blur or focus softness away from the key species cue in a realistic way",
    "mild backlight, shadow patching, or uneven natural illumination while species identity stays readable",
]

ANIMAL_CATALOG_STATE_VARIANTS = [
    "animal in a natural resting or still pose",
    "animal in a natural alert pose",
    "animal shown gently turning or reoriented",
    "animal shown in an unposed everyday moment",
    "natural body posture variation while key species-defining traits remain visible",
    "animal shown as one important element within a broader habitat view",
    "animal shown from a less idealized ordinary sighting angle",
    "animal engaged with the surrounding habitat in a natural way",
    "animal visible in a believable moment that feels observed rather than staged",
    "animal partially screened by ordinary habitat elements while remaining identifiable",
    "animal appearing as one strong cue within a busier real field observation",
]

ANIMAL_CATALOG_DISTRACTOR_VARIANTS = [
    "habitat elements such as grass, branches, rocks, or water can compete mildly for attention",
    "the animal may appear slightly smaller within a broader natural context while remaining identifiable",
    "minor partial overlap by foliage, water, or habitat clutter is allowed if species cues stay visible",
    "another non-dominant natural element may draw some attention without changing the labeled animal",
    "the animal need not be perfectly centered or isolated from the environment",
    "background activity or texture may be noticeable while the animal remains recognizable",
    "foreground vegetation, shadows, ripples, or branches may partially interfere without hiding species-defining cues",
    "other habitat textures and secondary natural elements may compete for attention like a real field sighting",
    "a nearby branch, rock, water reflection, or another habitat feature may draw comparable attention",
    "non-critical parts of the animal may sit near the image border in a realistic snapshot composition",
]

SCENE_CATALOG_VIEW_VARIANTS = [
    "slightly different composition",
    "slightly wider framing",
    "slightly tighter crop emphasizing the main subject",
    "slightly different vantage point",
    "main semantic region slightly off-center but clearly recognizable",
    "more diagonal composition",
    "lower eye-level scene view",
    "higher vantage scene view",
    "wider establishing view from farther away",
    "more incidental pass-by viewpoint rather than a centered textbook view",
    "main place cue read more from the mid-ground or background than a centered foreground feature",
]

SCENE_CATALOG_CONTEXT_VARIANTS = [
    "foreground and background layering remain visible",
    "moderate natural scene detail without excessive clutter",
    "clear place or structure cues with some surrounding context",
    "slightly varied framing of the main region",
    "depth cues from near and far elements",
    "more open surrounding space visible around the main semantic region",
    "nearby foreground element included without blocking the main scene cue",
    "broader environmental context included while place identity stays clear",
    "the key place cue occupies only part of a wider layout but remains recognizable",
    "the scene is read from multiple contextual elements rather than one isolated center cue",
    "foreground and side-edge elements take noticeable space while the place remains readable",
    "scene identity emerges through a crowded but coherent real-world layout",
]

SCENE_CATALOG_CAPTURE_VARIANTS = [
    "soft daylight variation",
    "light overcast or diffuse outdoor lighting",
    "ordinary documentary-style photo feel",
    "natural atmospheric contrast",
    "slight tonal variation while keeping the scene clear",
    "clear everyday travel-photo feel",
    "mild haze or humidity feel while main place cues remain clear",
    "late-afternoon-like natural light without dramatic stylization",
    "slightly imperfect travel or street snapshot feel with realistic exposure variation",
    "everyday observational photo rendering rather than a carefully composed postcard view",
    "mild consumer-camera limitations such as uneven sharpness or exposure are acceptable while place cues remain readable",
    "realistic weathering, haze, glass reflection, or shadow variation may affect parts of the frame",
]

SCENE_CATALOG_STATE_VARIANTS = [
    "scene appears quiet and unposed",
    "scene captured in an ordinary everyday moment",
    "scene shown with a more spacious establishing feel",
    "scene shown with stronger depth layering",
    "scene shown with a candid real-location feel",
    "scene shown with a busier everyday layout rather than a clean textbook view",
    "scene identity emerging from the full layout rather than one isolated cue",
    "scene identity conveyed through the overall arrangement rather than one dominant centered landmark",
    "scene feeling like a real encountered view rather than an ideal reference shot",
    "scene read as a plausible passing glimpse rather than a carefully selected postcard angle",
    "scene remaining semantically clear even though another region shares visual attention",
]

SCENE_CATALOG_DISTRACTOR_VARIANTS = [
    "foreground detail may compete mildly with the main scene cue",
    "secondary structures, vegetation, or objects may draw some attention while place identity stays clear",
    "the key semantic region need not dominate the entire frame as long as the place remains recognizable",
    "small distant human-made or natural details may be visible without becoming the main subject",
    "the scene can look mildly cluttered or imperfectly framed while staying semantically correct",
    "multiple non-dominant depth layers may share attention across the frame",
    "roads, railings, rocks, vegetation, signage, or incidental structures may share attention without changing the place identity",
    "a foreground edge occluder or side element may take noticeable space while the main scene remains readable",
    "a side object, railing, tree, wall, or foreground texture may momentarily compete with the main place cue",
    "the most salient visual patch need not be the semantic class cue as long as the whole scene remains recognizable",
]

OBJECT_CATALOG_SCENARIO_PROFILES = [
    "closer canonical real-world view, object relatively large in frame, frontal or near-frontal eye-level angle, off-center, everyday setting, one nearby object, surface edge, or foreground element occupies noticeable area and competes strongly for attention while key class-defining parts remain readable",
    "high-angle wider context view, clearly elevated or downward-looking viewpoint, object medium size in frame, surrounding room, workspace, shop, street, or storage context visible, several nearby objects or surfaces take substantial space and the object need not be the most salient region",
    "strong lateral side-view, clear side-profile or oblique side angle, object close to one frame edge, non-critical parts may approach the border, a nearby object, shelf edge, wall edge, or table edge occupies noticeable space and competes strongly for attention",
    "low-angle cluttered real-world view, clearly lower or upward-looking viewpoint, stronger depth perspective, multiple nearby objects, furniture, tools, packaging, or structural elements take substantial space, the object remains readable but competes with several strong visual elements",
    "imperfect off-axis snapshot, strong passing or turning viewpoint, partial foreground blockage, edge cutoff, uneven exposure, or focus falloff allowed, a foreground element or nearby object may take large visible area and rival the labeled object for attention while overall identity remains clear",
]

ANIMAL_CATALOG_SCENARIO_PROFILES = [
    "closer canonical animal view, animal relatively large in frame, frontal or near-frontal eye-level viewpoint, off-center, habitat visible, key species traits readable, foreground texture, vegetation, or another habitat element occupies noticeable area and competes strongly for attention",
    "high-angle wider habitat view, clearly elevated or downward-looking observational viewpoint, animal medium size in frame, broader environment visible, surrounding habitat elements take substantial space and the animal is not necessarily the strongest visual patch",
    "strong side-profile sighting, clear lateral side view rather than a frontal portrait, animal close to one frame edge, non-critical body parts may approach the border, side vegetation, rocks, water edges, or nearby habitat structure occupies noticeable space and competes strongly for attention",
    "low-angle busy habitat view, clearly lower or upward-looking ground-near viewpoint, stronger foreground-background layering, grass, branches, stones, water, or habitat clutter takes substantial space, the animal remains readable but competes with a visually busy environment",
    "imperfect off-axis animal snapshot, strong passing, turning, or oblique sighting viewpoint, partial screening, foreground obstruction, focus softness, or uneven exposure allowed, nearby habitat elements may take large visible area and rival the animal for attention while species identity remains clear",
]

SCENE_CATALOG_SCENARIO_PROFILES = [
    "closer canonical scene view, ordinary eye-level real-location viewpoint, main place or structure cue clearly readable, off-center, foreground and side elements occupy noticeable space and compete strongly for attention",
    "high-angle wider establishing view, clearly elevated or overlook-like viewpoint from farther away, the key place cue occupies only part of the layout, broader surroundings participate in recognition, secondary scene elements take substantial space and the main cue need not dominate the frame",
    "strong lateral or roadside scene view, diagonal side-view rather than a straight-on composition, main semantic region close to one side of the frame, side walls, rails, rocks, vegetation, or boundary structures occupy noticeable space and compete strongly for attention",
    "low-angle busy real-world scene, clearly lower or upward-emphasizing viewpoint, stronger depth perspective, layered foreground and background, rails, rocks, vegetation, signs, walls, or vehicles take substantial space while place identity stays clear, the most salient patch need not be the class cue",
    "imperfect off-axis scene snapshot, strong passing viewpoint, uneven exposure, haze, reflections, weathering, foreground edge clutter, or partial occluders allowed, non-central elements may take large visible area and rival the main place cue while the scene remains semantically clear",
]

CATALOG_NEGATIVE_PROMPT = (
    "illustration, cartoon, anime, painting, sketch, 3d render, cgi, blurry, low quality, noisy, "
    "jpeg artifacts, watermark, text, logo, multiple main subjects, duplicate subject, tiny distant subject, "
    "severe occlusion, underexposed, overexposed, harsh shadow, readable text, "
    "dramatic perspective, extreme camera angle, stylized lighting, oversaturated color"
)

SCENE_CATALOG_NEGATIVE_PROMPT_SUFFIX = (
    "portrait, selfie, close-up face, dominant foreground person, crowd"
)

PLACE_CATALOG_NEGATIVE_PROMPT_SUFFIX = (
    "customer, customers, barber cutting hair, butcher handling meat, cashier, shopkeeper, person behind counter"
)

# Short disambiguation cues for the few Tiny-ImageNet classes that are most likely
# to drift when only the bare class name is shown to the image model.
MINIMAL_CATALOG_CUES: Dict[str, str] = {
    "n02730930": "full apron garment clearly visible",
    "n09193705": "high mountain slope landscape",
    "n02699494": "religious altar without people",
    "n02788148": "stair handrail and balusters",
    "n02791270": "barbershop storefront or empty interior without people",
    "n02808440": "full-size bathtub clearly visible",
    "n02814860": "lighthouse beacon tower",
    "n02892201": "plaque",
    "n01917289": "brain coral colony",
    "n02906734": "natural straw broom",
    "n02927161": "butcher shop storefront or empty meat market without people",
    "n02988304": "full electronic CD player device with buttons and disc tray clearly visible, not a loose compact disc",
    "n02085620": "very small Chihuahua dog with large upright ears",
    "n03085013": "standalone full computer keyboard occupying most of the frame on a simple desk surface",
    "n03179701": "desk furniture as the main subject",
    "n03201208": "full dining table furniture, not a meal scene",
    "n03854065": "pipe organ",
    "n03126707": "construction crane",
    "n09246464": "steep rocky cliff face",
    "n03042490": "ancient stone dwelling built into a cliff",
    "n06596364": "closed comic book as the main object",
    "n03089624": "candy shop storefront or display without people",
    "n04067472": "spool",
    "n03160309": "large concrete dam structure",
    "n03544143": "glass sand timer with sand in two bulbs",
    "n02231487": "stick insect",
    "n07583066": "bowl of guacamole dip",
    "n09332890": "empty inland freshwater lakeshore with calm still water, no people, no ocean, no beach, no surf",
    "n03804744": "single large metal nail fastener as the main subject",
    "n04456115": "flaming torch or fire torch, not a flashlight",
    "n03891332": "single parking meter device",
    "n09428293": "coastline shoreline landscape without people",
    "n03976657": "utility pole",
    "n02099712": "short-haired Labrador retriever dog",
    "n02099601": "long-haired golden retriever dog with feathered golden coat",
    "n03014705": "full dresser furniture with drawers clearly visible",
    "n03980874": "full rain poncho waterproof outer garment clearly visible",
    "n04285008": "full sports car clearly visible",
    "n04356056": "full pair of sunglasses clearly visible",
    "n04417672": "broad full thatched roof section clearly visible",
    "n04532106": "full liturgical vestment robe clearly visible as the main subject",
    "n07579787": "round dinner plate",
    "n04008634": "metal projectile",
    "n03355925": "upright flagpole",
    "n04532670": "arched stone or concrete viaduct bridge",
}

EXTRA_CATALOG_GUIDANCE_BY_WNID: Dict[str, str] = {
    "n02808440": "full object visible, not cropped",
    "n02988304": "audio device itself is the dominant main subject, buttons and disc tray clearly visible, full device visible, not cropped, no loose disc as the main subject",
    "n03014705": "full furniture visible, not cropped",
    "n03085013": "keyboard alone dominates the image, keyboard fills most of the frame, full keyboard clearly visible, not cropped, on a simple desk surface, no person, no face, no hands, no headset, no monitor, no laptop, no other main objects",
    "n03179701": "desk itself is the main subject, any person must be secondary and not dominant",
    "n03201208": "table furniture itself is the main subject, no dominant meal",
    "n03544143": "timekeeping sand timer object clearly recognizable, not a person or body figure",
    "n03804744": "no hand holding it, no person",
    "n03854065": "musical instrument clearly recognizable, not anatomy",
    "n03980874": "poncho itself is the main subject, full poncho silhouette visible, not cropped, clearly a rain poncho, not a regular shirt or coat",
    "n04285008": "full vehicle visible, not cropped",
    "n04356056": "full object visible, not cropped",
    "n04417672": "large continuous thatched roof area clearly recognizable, not just a small edge or fragment",
    "n04532106": "vestment itself is the main subject, full ceremonial robe visible, a person may wear it but the garment must dominate, not a portrait crop",
    "n09332890": "inland freshwater lakeshore itself is the main subject, empty scene, no people, calm still water, inland setting, no surf, no ocean horizon",
    "n02730930": "apron itself is the dominant main subject, full garment visible, not cropped, simple kitchen or cooking setting",
}

EXTRA_CATALOG_NEGATIVE_BY_WNID: Dict[str, str] = {
    "n02808440": "sink, washbasin, basin",
    "n02906734": "green plastic broom, neon green broom",
    "n02988304": "loose compact disc only, single compact disc, cd disc only, jewel case, disc in hand, disc as the main subject",
    "n03014705": "cropped furniture, partial furniture",
    "n03085013": "cropped keyboard, partial keyboard, laptop only, person, face, portrait, headset, headphones, hands on keyboard, person using keyboard, portrait with keyboard, monitor as main subject, desk scene with person, keyboard as a secondary object, office scene as the main subject, computer setup as the main subject, mouse as the main subject",
    "n03179701": "person as the sole subject, office worker portrait, dominant foreground person",
    "n03201208": "meal scene, dinner plate as main subject, people dining",
    "n03544143": "person, human body, human figure, waist, torso, hourglass figure, body shape, silhouette, dress, gown, mannequin",
    "n03804744": "hand holding nail, person holding object",
    "n03854065": "human organ, anatomy, surgery, biology diagram",
    "n03980874": "blanket, shawl, regular shirt, coat, jacket, cropped garment, mannequin, hollow garment, floating garment, faceless void, fashion editorial, costume cape, glossy plastic costume",
    "n04285008": "cropped car, partial vehicle",
    "n04356056": "person wearing sunglasses, cropped eyewear",
    "n04417672": "cropped roof, partial roof edge only, tiny roof fragment",
    "n04532106": "cropped robe, portrait, face as the main subject, person wearing robe as portrait",
    "n04456115": "flashlight, electric torch",
    "n09332890": "person, people, human, figure in scene, person standing by the lake, beach scene, ocean coast, sea, seashore, coastline, surf, waves, tropical beach, sandy ocean shore, ocean horizon, palm trees, beach umbrellas",
    "n02730930": "fashion portrait, model wearing apron as the main subject, plain white studio backdrop",
}

CLASS_SPECIFIC_CATALOG_PROMPT_OVERRIDES: Dict[str, str] = {
    "n03085013": (
        "A clean reference natural photograph of computer keyboard, showing a standalone full computer keyboard on a simple "
        "desk surface, keyboard occupies most of the frame and is the only main subject, full keyboard clearly visible, "
        "not cropped, no person, no face, no hands, no monitor, no laptop, no other main objects, clear main subject, "
        "realistic real-world setting, clean composition, sharp focus, natural color, visually consistent reference-photo style."
    ),
    "n03980874": (
        "A realistic candid outdoor photograph of poncho, showing a person naturally wearing a full waterproof rain poncho "
        "in light rain, the poncho garment is the main subject, full poncho silhouette clearly visible, realistic human "
        "posture and proportions, natural folds and wet fabric texture, person is secondary, face visible but not "
        "emphasized, clearly a practical rain poncho and not a regular coat or shirt, full garment visible and not cropped, "
        "real outdoor rainy setting, clean composition, sharp focus, natural color, visually consistent reference-photo style."
    ),
    "n04417672": (
        "A clean reference natural photograph of thatch, showing a traditional hut or house with a full thatched roof "
        "clearly visible, the thatched roof is the main subject, most of the roof visible in frame, roof material and "
        "overall roof form clearly recognizable, clear main subject, realistic real-world setting, clean composition, "
        "sharp focus, natural color, visually consistent reference-photo style."
    ),
    "n09332890": (
        "A clean reference natural photograph of lakeside, showing an inland freshwater lakeshore with calm still lake "
        "water, grassy or rocky shoreline, and distant treeline on the opposite shore, empty scene with no people, "
        "no ocean, no sea, no beach, no surf, clear main subject, realistic real-world setting, clean composition, "
        "sharp focus, natural color, visually consistent reference-photo style."
    ),
}

SCENE_LIKE_WNIDS = {
    "n09193705",  # alp
    "n02699494",  # altar
    "n02788148",  # bannister
    "n02791270",  # barbershop
    "n02814860",  # beacon
    "n01917289",  # brain coral
    "n02927161",  # butcher shop
    "n09246464",  # cliff
    "n03042490",  # cliff dwelling
    "n03089624",  # confectionery
    "n03160309",  # dam
    "n09332890",  # lakeside
    "n09428293",  # seashore
    "n04532670",  # viaduct
}

LANDSCAPE_WNIDS = {
    "n09193705",  # alp
    "n01917289",  # brain coral
    "n09246464",  # cliff
    "n09332890",  # lakeside
    "n09428293",  # seashore
}

PLACE_WNIDS = {
    "n02699494",  # altar
    "n02791270",  # barbershop
    "n02927161",  # butcher shop
    "n03089624",  # confectionery
}

STRUCTURE_WNIDS = {
    "n02814860",  # beacon
    "n03042490",  # cliff dwelling
    "n03160309",  # dam
    "n04532670",  # viaduct
}

DETAIL_WNIDS = {
    "n02788148",  # bannister
}

ANIMAL_VARIANT_WNIDS = {
    "n01443537",  # goldfish
    "n01629819",  # European fire salamander
    "n01641577",  # bullfrog
    "n01644900",  # tailed frog
    "n01698640",  # American alligator
    "n01742172",  # boa constrictor
    "n01770393",  # scorpion
    "n01774384",  # black widow
    "n01774750",  # tarantula
    "n01784675",  # centipede
    "n01855672",  # goose
    "n01944390",  # snail
    "n01945685",  # slug
    "n01950731",  # sea slug
    "n01983481",  # American lobster
    "n01984695",  # spiny lobster
    "n02002724",  # black stork
    "n02056570",  # king penguin
    "n02058221",  # albatross
    "n02085620",  # Chihuahua
    "n02094433",  # Yorkshire terrier
    "n02099601",  # golden retriever
    "n02099712",  # Labrador retriever
    "n02106662",  # German shepherd
    "n02113799",  # standard poodle
    "n02123045",  # tabby
    "n02123394",  # Persian cat
    "n02124075",  # Egyptian cat
    "n02125311",  # cougar
    "n02129165",  # lion
    "n02132136",  # brown bear
    "n02165456",  # ladybug
    "n02190166",  # fly
    "n02206856",  # bee
    "n02226429",  # grasshopper
    "n02231487",  # walking stick
    "n02233338",  # cockroach
    "n02236044",  # mantis
    "n02268443",  # dragonfly
    "n02279972",  # monarch
    "n02281406",  # sulphur butterfly
    "n02321529",  # sea cucumber
    "n02364673",  # guinea pig
    "n02395406",  # hog
    "n02403003",  # ox
    "n02410509",  # bison
    "n02415577",  # bighorn
    "n02423022",  # gazelle
    "n02437312",  # Arabian camel
    "n02480495",  # orangutan
    "n02481823",  # chimpanzee
    "n02486410",  # baboon
    "n02504458",  # African elephant
    "n02509815",  # lesser panda
    "n01910747",  # jellyfish
    "n02074367",  # dugong
    "n01768244",  # trilobite
}

PROMPT_TEMPLATE = (
    "A realistic color photo of {class_name}, {scene_hint}, "
    + COMPOSITION_ANCHOR
    + ", "
    + LIGHTING_ANCHOR
    + ", "
    + STYLE_ANCHOR
    + "."
)


def extract_first_image_url(payload) -> Optional[str]:
    """Try to find the first generated image URL in a DashScope response payload."""

    urls: List[str] = []

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, str):
            if node.startswith("http://") or node.startswith("https://"):
                urls.append(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if hasattr(node, "items"):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in {"image", "image_url", "url"}:
                    if isinstance(value, str):
                        if value.startswith("http://") or value.startswith("https://"):
                            urls.append(value)
                    elif isinstance(value, dict):
                        walk(value.get("url"))
                    else:
                        walk(value)
                else:
                    walk(value)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in {"image", "image_url", "url"}:
                    if isinstance(value, str):
                        if value.startswith("http://") or value.startswith("https://"):
                            urls.append(value)
                    elif isinstance(value, dict):
                        walk(value.get("url"))
                    else:
                        walk(value)
                else:
                    walk(value)
            return
        if hasattr(node, "__dict__"):
            walk(vars(node))

    walk(payload)
    return urls[0] if urls else None


def response_status_code(response) -> Optional[int]:
    if hasattr(response, "status_code"):
        return int(response.status_code)
    if isinstance(response, dict) and "status_code" in response:
        return int(response["status_code"])
    return None


def response_error_text(response) -> str:
    code = getattr(response, "code", None)
    if code is None and isinstance(response, dict):
        code = response.get("code")
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    if isinstance(response, dict) and isinstance(response.get("error"), dict):
        error_info = response["error"]
        code = code or error_info.get("code")
        message = message or error_info.get("message")
    parts = []
    if code:
        parts.append(f"code={code}")
    if message:
        parts.append(f"message={message}")
    return ", ".join(parts) if parts else "unknown error"


def download_image(url: str, output_path: Path, timeout: int = 60) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r:  # nosec B310 - generated asset URL
        output_path.write_bytes(r.read())


def write_image_bytes(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)


def open_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def center_crop_resize_square(image: Image.Image, size: int, crop_frac: float = 1.0) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    side = min(w, h)
    crop_frac = max(0.1, min(float(crop_frac), 1.0))
    side = max(1, int(round(side * crop_frac)))
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def sharpen_small_image(
    image: Image.Image,
    radius: float = 0.8,
    percent: int = 140,
    threshold: int = 2,
) -> Image.Image:
    # Mild sharpening to recover edge contrast after aggressive downsampling to 64x64.
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def _clean_piece(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.strip(" ,.")


def _unique_pieces(parts: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in parts:
        piece = _clean_piece(raw)
        if not piece:
            continue
        key = piece.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(piece)
    return out


def render_framing_hint(framing_hint: str) -> str:
    hint = framing_hint.strip().lower()
    mapping = {
        "close_up": "close-up view with key parts large and clearly visible",
        "mid_shot": "mid shot with one clear subject and limited background distraction",
        "full_object": "full subject visible in frame, centered, not cut off",
    }
    return mapping.get(hint, hint)


def build_prompt(class_name: str, scene_hint: str) -> str:
    return PROMPT_TEMPLATE.format(class_name=class_name, scene_hint=scene_hint)


def _normalize_catalog_cue(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).strip(" ,.;:").lower()


def choose_catalog_cue(
    rec,
    semantic_row: Optional[Dict],
    max_include_keywords: int = 1,
) -> Tuple[Optional[str], Optional[str]]:
    cue = MINIMAL_CATALOG_CUES.get(rec.wnid)
    if cue:
        return cue, "minimal_catalog_cues"

    if semantic_row and max_include_keywords > 0:
        include_keywords = [str(x).strip() for x in semantic_row.get("include_keywords", []) if str(x).strip()]
        rec_name_norm = _normalize_catalog_cue(rec.name)
        rec_name_tokens = set(rec_name_norm.split())
        for keyword in include_keywords[:max_include_keywords]:
            keyword_norm = _normalize_catalog_cue(keyword)
            keyword_tokens = set(keyword_norm.split())
            if keyword_norm == rec_name_norm:
                continue
            # Skip weak cues that are already just a subset of the class name,
            # e.g. "egyptian" for "Egyptian cat".
            if keyword_norm in rec_name_norm:
                continue
            if keyword_tokens and keyword_tokens.issubset(rec_name_tokens):
                continue
            return keyword, "semantic_include_keywords"

    return None, None


def get_catalog_extra_guidance(rec) -> List[str]:
    pieces: List[str] = []
    if rec.wnid in SCENE_LIKE_WNIDS:
        pieces.append(SCENE_REFERENCE_PROMPT_ADDON)
    if rec.wnid in PLACE_WNIDS:
        pieces.append(PLACE_REFERENCE_PROMPT_ADDON)
    extra = EXTRA_CATALOG_GUIDANCE_BY_WNID.get(rec.wnid)
    if extra:
        pieces.append(extra)
    return pieces


def choose_catalog_sample_variant(rec, sample_idx: int, total_samples: int) -> Optional[str]:
    if total_samples <= 1:
        return None

    if rec.wnid in ANIMAL_VARIANT_WNIDS:
        scenario_profiles = ANIMAL_CATALOG_SCENARIO_PROFILES
    elif rec.wnid in SCENE_LIKE_WNIDS:
        scenario_profiles = SCENE_CATALOG_SCENARIO_PROFILES
    else:
        scenario_profiles = OBJECT_CATALOG_SCENARIO_PROFILES

    if not scenario_profiles:
        return None

    slot = (sample_idx - 1) % len(scenario_profiles)
    return scenario_profiles[slot]


def build_catalog_negative_prompt(rec, semantic_row: Optional[Dict]) -> str:
    parts = [CATALOG_NEGATIVE_PROMPT]
    if rec.wnid in SCENE_LIKE_WNIDS:
        parts.append(SCENE_CATALOG_NEGATIVE_PROMPT_SUFFIX)
    if rec.wnid in PLACE_WNIDS:
        parts.append(PLACE_CATALOG_NEGATIVE_PROMPT_SUFFIX)
    extra = EXTRA_CATALOG_NEGATIVE_BY_WNID.get(rec.wnid)
    if extra:
        parts.append(extra)
    return ", ".join(_unique_pieces(parts))


def build_catalog_prompt(
    rec,
    semantic_row: Optional[Dict],
    max_include_keywords: int = 1,
    sample_idx: int = 1,
    total_samples: int = 1,
    use_sample_variation: bool = True,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    if rec.wnid in CLASS_SPECIFIC_CATALOG_PROMPT_OVERRIDES:
        base_prompt = _clean_piece(CLASS_SPECIFIC_CATALOG_PROMPT_OVERRIDES[rec.wnid]).rstrip(".")
        sample_variant = None
        if use_sample_variation:
            sample_variant = choose_catalog_sample_variant(rec, sample_idx, total_samples)
        if sample_variant:
            merged = ", ".join(_unique_pieces([base_prompt, sample_variant]))
            return merged + ".", None, "class_specific_catalog_prompt_override", sample_variant
        return base_prompt + ".", None, "class_specific_catalog_prompt_override", None

    pieces = [f"A clean reference natural photograph of {rec.name}"]
    cue, cue_source = choose_catalog_cue(rec, semantic_row, max_include_keywords=max_include_keywords)
    if cue:
        pieces.append(f"showing {cue}")
    pieces.extend(get_catalog_extra_guidance(rec))
    sample_variant = None
    if use_sample_variation:
        sample_variant = choose_catalog_sample_variant(rec, sample_idx, total_samples)
        if sample_variant:
            pieces.append(sample_variant)
    pieces.append(REFERENCE_PHOTO_STYLE_PROMPT_SUFFIX)
    return ", ".join(_unique_pieces(pieces)) + ".", cue, cue_source, sample_variant


def build_disambiguated_prompt(
    rec,
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

    pieces = [f"A realistic color photo of {rec.name}"]
    if sense:
        pieces.append(f"intended Tiny-ImageNet sense: {sense}")
    if include_keywords:
        pieces.append("key visual cues: " + ", ".join(include_keywords[:max_include_keywords]))
    pieces.append(scene_override or scene_hint)
    if framing_hint:
        pieces.append(render_framing_hint(framing_hint))
    pieces.extend([COMPOSITION_ANCHOR, LIGHTING_ANCHOR, STYLE_ANCHOR])
    return ", ".join(_unique_pieces(pieces)) + "."


def build_disambiguated_negative_prompt(semantic_row: Optional[Dict], max_exclude_keywords: int = 3) -> str:
    parts = [NEGATIVE_PROMPT]
    if semantic_row:
        exclude_keywords = [str(x).strip() for x in semantic_row.get("exclude_keywords", []) if str(x).strip()]
        if exclude_keywords:
            parts.append(", ".join(exclude_keywords[:max_exclude_keywords]))
    return ", ".join(_unique_pieces(parts))


def choose_sample_seed(base_seed: int, class_index: int, sample_idx: int) -> int:
    rng = random.Random(base_seed + class_index * 100003 + sample_idx * 97)
    return rng.randrange(1, 2**31 - 1)


def maybe_load_response_json(response) -> Dict:
    if isinstance(response, dict):
        return response
    try:
        return json.loads(json.dumps(response, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        out = {}
        if hasattr(response, "output"):
            out["output"] = getattr(response, "output")
        if hasattr(response, "usage"):
            out["usage"] = getattr(response, "usage")
        if hasattr(response, "request_id"):
            out["request_id"] = getattr(response, "request_id")
        if hasattr(response, "status_code"):
            out["status_code"] = getattr(response, "status_code")
        if hasattr(response, "code"):
            out["code"] = getattr(response, "code")
        if hasattr(response, "message"):
            out["message"] = getattr(response, "message")
        if hasattr(response, "__dict__"):
            for key, value in vars(response).items():
                out.setdefault(key, value)
        return out


def normalize_size(size: str, style: str) -> str:
    text = size.strip().lower().replace("*", "x")
    if "x" not in text:
        raise ValueError(f"Invalid size format: {size}. Use forms like 1024x1024 or 1024*1024.")
    width, height = [x.strip() for x in text.split("x", 1)]
    if not width.isdigit() or not height.isdigit():
        raise ValueError(f"Invalid size format: {size}. Width and height must be integers.")
    if style == "dashscope":
        return f"{width}*{height}"
    return f"{width}x{height}"


def use_compatible_mode(base_url: str) -> bool:
    return "compatible-mode" in base_url


def is_qwen_image_model(model: str) -> bool:
    return model.strip().lower().startswith("qwen-image")


def convert_compatible_to_dashscope_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    replacements = {
        "https://dashscope.aliyuncs.com/compatible-mode/v1": "https://dashscope.aliyuncs.com/api/v1",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1": "https://dashscope-intl.aliyuncs.com/api/v1",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1": "https://dashscope-us.aliyuncs.com/api/v1",
    }
    return replacements.get(url, url)


def call_compatible_image_generation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    negative_prompt: Optional[str],
    size: str,
) -> Dict:
    endpoint = base_url.rstrip("/") + "/images/generations"
    payload = {
        "model": model,
        "prompt": prompt,
        "size": normalize_size(size, style="compatible"),
        "n": 1,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310 - trusted user-specified API endpoint
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                parsed.setdefault("status_code", getattr(response, "status", 200))
            return parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"message": body}
        if isinstance(parsed, dict):
            parsed.setdefault("status_code", e.code)
            parsed.setdefault("message", body or f"HTTP {e.code}")
        return parsed


def call_dashscope_multimodal_generation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    negative_prompt: Optional[str],
    size: str,
    watermark: bool,
    prompt_extend: bool,
) -> Dict:
    endpoint = base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation"
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
            "size": normalize_size(size, style="dashscope"),
        },
    }
    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310 - trusted user-specified API endpoint
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                parsed.setdefault("status_code", getattr(response, "status", 200))
            return parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"message": body}
        if isinstance(parsed, dict):
            parsed.setdefault("status_code", e.code)
            parsed.setdefault("message", body or f"HTTP {e.code}")
        return parsed


def extract_first_image_b64(payload) -> Optional[bytes]:
    blobs: List[bytes] = []

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if hasattr(node, "items"):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower == "b64_json" and isinstance(value, str):
                    try:
                        blobs.append(base64.b64decode(value))
                    except Exception:  # noqa: BLE001
                        continue
                else:
                    walk(value)
            return
        if hasattr(node, "__dict__"):
            walk(vars(node))

    walk(payload)
    return blobs[0] if blobs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tiny target-domain data with Qwen Image API")
    parser.add_argument(
        "--class-map",
        type=str,
        default="./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
        help="Path to Tiny-ImageNet class map JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/tiny-target-domain-qwen-5shot",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--semantic-spec",
        type=str,
        default=None,
        help="Optional semantic spec JSON for disambiguation",
    )
    parser.add_argument(
        "--strict-disambiguation",
        action="store_true",
        help="When set, force include/exclude semantic keywords into prompt construction",
    )
    parser.add_argument("--max-include-keywords", type=int, default=3)
    parser.add_argument("--max-exclude-keywords", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=5, help="How many images to generate per class")
    parser.add_argument("--base-seed", type=int, default=20260418, help="Only used for manifest bookkeeping")
    parser.add_argument("--start-index", type=int, default=0, help="Start class index for partial generation")
    parser.add_argument("--end-index", type=int, default=None, help="End class index (exclusive) for partial generation")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for API or download failures")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose 64x64 image already exists")
    parser.add_argument("--dry-run", action="store_true", help="Only print prompts, do not call API")
    parser.add_argument("--model", type=str, default="qwen-image-2.0")
    parser.add_argument("--size", type=str, default="2048x2048", help="Image size, e.g. 1024x1024 or 2048*2048")
    parser.add_argument("--watermark", action="store_true", help="Enable watermark. Default is off.")
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["descriptive", "catalog"],
        default="descriptive",
        help="Prompt style. descriptive = current long semantic prompt; catalog = stronger unified studio/catalog style.",
    )
    parser.add_argument(
        "--catalog-include-keywords",
        type=int,
        default=1,
        help="When --prompt-style=catalog, keep at most this many semantic cue keywords.",
    )
    parser.add_argument(
        "--disable-catalog-sample-variation",
        action="store_true",
        help="Disable the lightweight per-sample prompt variation used to diversify multi-shot catalog generation.",
    )
    parser.add_argument(
        "--crop-frac",
        type=float,
        default=1.0,
        help="Center-crop fraction before resizing. <1.0 zooms in and reduces background, e.g. 0.85.",
    )
    parser.add_argument(
        "--use-sharpened-output",
        action="store_true",
        help="Use the sharpened 64x64 image as the main output file under images/.",
    )
    parser.add_argument(
        "--only-raw-and-64",
        action="store_true",
        help="Only save raw_images/ and images/64x64 outputs. Do not create images_128/ or images_64_sharpened/.",
    )
    parser.add_argument(
        "--raw-inside-images",
        action="store_true",
        help="When used with --only-raw-and-64, save both raw and 64x64 files inside images/<class>/ as raw_image_XXXX.png and image_XXXX.png.",
    )
    parser.add_argument("--sharpen-radius", type=float, default=0.8)
    parser.add_argument("--sharpen-percent", type=int, default=140)
    parser.add_argument("--sharpen-threshold", type=int, default=2)
    parser.set_defaults(prompt_extend=True)
    parser.add_argument(
        "--prompt-extend",
        dest="prompt_extend",
        action="store_true",
        help="Enable prompt extension (default: enabled)",
    )
    parser.add_argument(
        "--disable-prompt-extend",
        dest="prompt_extend",
        action="store_false",
        help="Disable prompt extension",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="API base URL. Compatible mode example: https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key", type=str, default=None, help="DashScope API key. Default: read env var")
    parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY", help="Env var name for API key")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional global negative prompt override. If unset, use script default / semantic negative prompt.",
    )
    args = parser.parse_args()

    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1")

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key and not args.dry_run:
        raise ValueError(
            f"Missing API key. Set --api-key or export {args.api_key_env}=sk-xxx before running."
        )

    requested_base_url = args.base_url
    compatible_mode = use_compatible_mode(args.base_url)
    qwen_image_mode = is_qwen_image_model(args.model)
    effective_base_url = args.base_url
    api_mode = "compatible" if compatible_mode else "dashscope"

    if qwen_image_mode:
        effective_base_url = convert_compatible_to_dashscope_base_url(args.base_url)
        api_mode = "dashscope_multimodal"
        if compatible_mode:
            print(
                "[WARN] qwen-image models use the DashScope multimodal-generation endpoint; "
                f"switching base_url to {effective_base_url}"
            )

    class_map_path = Path(args.class_map).resolve()
    if not class_map_path.exists():
        raise FileNotFoundError(
            "class map not found: "
            f"{class_map_path}. "
            "Build it first with scripts/build_tiny_imagenet_class_map.py."
        )

    out_root = Path(args.output_dir).resolve()
    images_root = out_root / "images"
    images_128_root = out_root / "images_128"
    images_64_sharpened_root = out_root / "images_64_sharpened"
    raw_root = out_root / "raw_images"
    inline_raw_inside_images = bool(args.raw_inside_images and args.only_raw_and_64)
    images_root.mkdir(parents=True, exist_ok=True)
    if not args.only_raw_and_64:
        images_128_root.mkdir(parents=True, exist_ok=True)
        images_64_sharpened_root.mkdir(parents=True, exist_ok=True)
    if not inline_raw_inside_images:
        raw_root.mkdir(parents=True, exist_ok=True)

    classes = load_class_map(class_map_path)
    if args.end_index is None:
        end_index = len(classes)
    else:
        end_index = min(args.end_index, len(classes))
    classes = classes[args.start_index:end_index]

    semantic_path = Path(args.semantic_spec).resolve() if args.semantic_spec else None
    if semantic_path is not None and not semantic_path.exists():
        raise FileNotFoundError(
            "semantic spec not found: "
            f"{semantic_path}. "
            "Build it first with scripts/build_tiny_semantic_review_list.py."
        )
    semantic_table = load_semantic_spec(semantic_path)
    if args.strict_disambiguation and semantic_path is None:
        print("[WARN] strict disambiguation enabled without semantic spec; fallback to base prompt")

    used_dir_names: Dict[str, int] = {}
    manifest_path = out_root / "manifest_qwen_api.jsonl"
    failed_path = out_root / "failed_cases_qwen_api.json"
    missing_semantics_path = out_root / "missing_semantics_qwen_api.json"

    failed_cases: List[Dict] = []
    missing_semantics: List[Dict] = []

    prompt_extend = bool(args.prompt_extend)

    print(f"[CONFIG] model={args.model}")
    print(f"[CONFIG] requested_base_url={requested_base_url}")
    print(f"[CONFIG] effective_base_url={effective_base_url}")
    print(f"[CONFIG] api_mode={api_mode}")
    print(f"[CONFIG] output_dir={out_root}")
    print(f"[CONFIG] class_count={len(classes)}")
    print(f"[CONFIG] prompt_style={args.prompt_style}")
    print(f"[CONFIG] catalog_sample_variation={not bool(args.disable_catalog_sample_variation)}")
    print(f"[CONFIG] crop_frac={args.crop_frac}")
    print(f"[CONFIG] use_sharpened_output={bool(args.use_sharpened_output)}")
    print(f"[CONFIG] only_raw_and_64={bool(args.only_raw_and_64)}")
    print(f"[CONFIG] raw_inside_images={bool(args.raw_inside_images)}")

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rec in classes:
            base_dir_name = sanitize_class_dir_name(rec.name)
            count = used_dir_names.get(base_dir_name, 0)
            used_dir_names[base_dir_name] = count + 1
            class_dir_name = base_dir_name if count == 0 else f"{base_dir_name}_{rec.wnid}"

            class_img_dir = images_root / class_dir_name
            class_img_128_dir = images_128_root / class_dir_name
            class_img_64_sharpened_dir = images_64_sharpened_root / class_dir_name
            class_raw_dir = raw_root / class_dir_name
            class_img_dir.mkdir(parents=True, exist_ok=True)
            if not args.only_raw_and_64:
                class_img_128_dir.mkdir(parents=True, exist_ok=True)
                class_img_64_sharpened_dir.mkdir(parents=True, exist_ok=True)
            if not inline_raw_inside_images:
                class_raw_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx in range(1, args.samples_per_class + 1):
                if inline_raw_inside_images:
                    out_img = class_img_dir / f"image_{sample_idx:04d}.png"
                    out_raw = class_img_dir / f"raw_image_{sample_idx:04d}.png"
                else:
                    out_img = class_img_dir / f"{sample_idx:04d}.png"
                    out_raw = class_raw_dir / f"{sample_idx:04d}.png"
                out_img_128 = class_img_128_dir / f"{sample_idx:04d}.png"
                out_img_64_sharpened = class_img_64_sharpened_dir / f"{sample_idx:04d}.png"

                if args.skip_existing and out_img.exists():
                    row = {
                        "index": rec.index,
                        "wnid": rec.wnid,
                        "class_name": rec.name,
                        "class_dir_name": class_dir_name,
                        "sample_idx": sample_idx,
                        "status": "skipped_existing",
                        "output_path": str(out_img),
                        "output_path_128": None if args.only_raw_and_64 else str(out_img_128),
                        "output_path_64_sharpened": None if args.only_raw_and_64 else str(out_img_64_sharpened),
                        "raw_output_path": str(out_raw),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue

                bookkeeping_seed = choose_sample_seed(args.base_seed, rec.index, sample_idx)
                scene_hint = choose_scene_hint(args.base_seed + sample_idx, rec.index)
                semantic_row = semantic_table.get(rec.wnid)
                catalog_cue = None
                catalog_cue_source = None
                catalog_sample_variant = None

                if args.prompt_style == "catalog":
                    prompt, catalog_cue, catalog_cue_source, catalog_sample_variant = build_catalog_prompt(
                        rec,
                        semantic_row,
                        max_include_keywords=max(0, args.catalog_include_keywords),
                        sample_idx=sample_idx,
                        total_samples=args.samples_per_class,
                        use_sample_variation=not bool(args.disable_catalog_sample_variation),
                    )
                    negative_prompt = (
                        args.negative_prompt
                        if args.negative_prompt is not None
                        else build_catalog_negative_prompt(rec, semantic_row)
                    )
                else:
                    if semantic_row:
                        prompt = build_disambiguated_prompt(
                            rec,
                            scene_hint,
                            semantic_row,
                            max_include_keywords=max(0, args.max_include_keywords),
                        )
                        negative_prompt = (
                            args.negative_prompt
                            if args.negative_prompt is not None
                            else build_disambiguated_negative_prompt(
                                semantic_row,
                                max_exclude_keywords=max(0, args.max_exclude_keywords),
                            )
                        )
                    else:
                        prompt = build_prompt(rec.name, scene_hint)
                        negative_prompt = args.negative_prompt if args.negative_prompt is not None else NEGATIVE_PROMPT
                        if semantic_path is not None:
                            missing_semantics.append({"wnid": rec.wnid, "name": rec.name})

                row = {
                    "index": rec.index,
                    "wnid": rec.wnid,
                    "class_name": rec.name,
                    "class_dir_name": class_dir_name,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "catalog_cue": catalog_cue,
                    "catalog_cue_source": catalog_cue_source,
                    "catalog_sample_variant": catalog_sample_variant,
                    "semantic_sense": (semantic_row or {}).get("sense"),
                    "semantic_include_keywords": (semantic_row or {}).get("include_keywords", []),
                    "semantic_exclude_keywords": (semantic_row or {}).get("exclude_keywords", []),
                    "bookkeeping_seed": bookkeeping_seed,
                    "model_id": args.model,
                    "prompt_style": args.prompt_style,
                    "requested_base_url": requested_base_url,
                    "base_url": effective_base_url,
                    "api_mode": api_mode,
                    "size": args.size,
                    "watermark": bool(args.watermark),
                    "prompt_extend": bool(prompt_extend),
                    "status": "ok",
                    "output_path": str(out_img),
                    "output_path_128": None if args.only_raw_and_64 else str(out_img_128),
                    "output_path_64_sharpened": None if args.only_raw_and_64 else str(out_img_64_sharpened),
                    "raw_output_path": str(out_raw),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }

                if args.dry_run:
                    print(f"\n[{rec.index:03d}] {rec.name} / sample {sample_idx}")
                    print(prompt)
                    print("NEG:", negative_prompt)
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue

                success = False
                for attempt in range(args.max_retries + 1):
                    try:
                        if api_mode == "compatible":
                            response = call_compatible_image_generation(
                                api_key=api_key,
                                base_url=effective_base_url,
                                model=args.model,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                size=args.size,
                            )
                        else:
                            response = call_dashscope_multimodal_generation(
                                api_key=api_key,
                                base_url=effective_base_url,
                                model=args.model,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                size=args.size,
                                watermark=bool(args.watermark),
                                prompt_extend=bool(prompt_extend),
                            )

                        row["status_code"] = response_status_code(response)
                        payload = maybe_load_response_json(response)
                        row["response_request_id"] = payload.get("request_id")
                        if response_status_code(response) != 200:
                            row["response_payload"] = payload

                        if response_status_code(response) != 200:
                            raise RuntimeError(response_error_text(response))

                        image_url = extract_first_image_url(payload)
                        image_b64 = extract_first_image_b64(payload)
                        if image_url:
                            row["image_url"] = image_url
                            download_image(image_url, out_raw)
                        elif image_b64:
                            row["image_source"] = "b64_json"
                            write_image_bytes(image_b64, out_raw)
                        else:
                            raise RuntimeError("no image asset found in API response")

                        img = open_image_rgb(out_raw)
                        final_img_64 = center_crop_resize_square(img, 64, crop_frac=args.crop_frac)
                        if args.only_raw_and_64:
                            final_img_64.save(out_img)
                        else:
                            debug_img_128 = center_crop_resize_square(img, 128, crop_frac=args.crop_frac)
                            final_img_64_sharpened = sharpen_small_image(
                                final_img_64,
                                radius=args.sharpen_radius,
                                percent=args.sharpen_percent,
                                threshold=args.sharpen_threshold,
                            )
                            debug_img_128.save(out_img_128)
                            if args.use_sharpened_output:
                                final_img_64_sharpened.save(out_img)
                            else:
                                final_img_64.save(out_img)
                            final_img_64_sharpened.save(out_img_64_sharpened)
                        row["crop_frac"] = args.crop_frac
                        row["use_sharpened_output"] = bool(args.use_sharpened_output) and not bool(args.only_raw_and_64)
                        row["sharpen_radius"] = None if args.only_raw_and_64 else args.sharpen_radius
                        row["sharpen_percent"] = None if args.only_raw_and_64 else args.sharpen_percent
                        row["sharpen_threshold"] = None if args.only_raw_and_64 else args.sharpen_threshold
                        success = True
                        break

                    except Exception as e:  # noqa: BLE001
                        row["status"] = "failed"
                        row["error"] = str(e)
                        row["attempt"] = attempt + 1
                        if "Connection refused" in str(e):
                            row["error_hint"] = (
                                "Cannot reach the remote API endpoint. "
                                "Check outbound network access, proxy settings, and base_url."
                            )
                        if attempt >= args.max_retries:
                            failed_cases.append(
                                {
                                    "index": rec.index,
                                    "wnid": rec.wnid,
                                    "class_name": rec.name,
                                    "sample_idx": sample_idx,
                                    "status_code": row.get("status_code"),
                                    "reason": str(e),
                                    "response_payload": row.get("response_payload"),
                                    "prompt": prompt,
                                }
                            )
                        else:
                            time.sleep(min(2.0, args.sleep_seconds * (attempt + 1)))

                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[{'OK' if success else 'FAIL'}] "
                    f"class={rec.index:03d} {rec.name} sample={sample_idx:02d}"
                )
                time.sleep(args.sleep_seconds)

    unique_missing = sorted({(x["wnid"], x["name"]) for x in missing_semantics})
    missing_rows = [{"wnid": w, "name": n} for w, n in unique_missing]
    failed_path.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_semantics_path.write_text(json.dumps(missing_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] failed_cases: {failed_path} ({len(failed_cases)})")
    print(f"[OK] missing_semantics: {missing_semantics_path} ({len(missing_rows)})")
    print(f"[OK] images root: {images_root}")
    if not args.only_raw_and_64:
        print(f"[OK] images_128 root: {images_128_root}")
        print(f"[OK] images_64_sharpened root: {images_64_sharpened_root}")
    if inline_raw_inside_images:
        print("[OK] raw images are stored inside each class directory under images/")
    else:
        print(f"[OK] raw images root: {raw_root}")


if __name__ == "__main__":
    main()
