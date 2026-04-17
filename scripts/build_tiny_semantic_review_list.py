#!/usr/bin/env python3
"""Build Tiny-ImageNet semantic spec and ambiguity risk reports."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


AMBIGUOUS_TOKENS = {
    "reel", "crane", "organ", "pole", "slug", "seal", "bass", "monitor",
    "plate", "tank", "torch", "trunk", "tick", "web", "cardigan", "pier",
    "maillot", "pan", "file", "drum", "bar", "bow", "mole", "sock", "jaguar", "plunger",
}


DISAMBIG_HINTS: Dict[str, Dict] = {
    "n04067472": {
        "sense": "winding reel or spool object used for cable, film, or hose",
        "include_keywords": ["spool", "cylindrical core", "wound cable", "industrial object"],
        "exclude_keywords": ["movie scene", "film director", "social media reel", "human portrait"],
        "scene_hint_override": "object product photo on workbench, clear structure visible",
        "framing_hint": "close_up",
    },
    "n03126707": {
        "sense": "construction crane machine used at building sites",
        "include_keywords": ["tower crane", "steel boom", "construction site", "heavy machinery"],
        "exclude_keywords": ["bird", "heron", "wildlife", "feathers"],
        "scene_hint_override": "urban construction environment with concrete and steel",
        "framing_hint": "full_object",
    },
    "n03854065": {
        "sense": "pipe organ musical instrument with keyboard and pipes",
        "include_keywords": ["pipe organ", "keyboard", "church interior", "instrument"],
        "exclude_keywords": ["human anatomy", "medical organ", "surgery", "biology diagram"],
        "scene_hint_override": "church or concert hall interior with visible pipes",
        "framing_hint": "mid_shot",
    },
    "n03976657": {
        "sense": "plain straight pole object, not decorative maypole or flagpole",
        "include_keywords": ["straight pole", "simple rod", "utility support"],
        "exclude_keywords": ["maypole ribbons", "national flag", "festival dance", "ornamental pole"],
        "scene_hint_override": "minimal outdoor or workshop context with isolated pole",
        "framing_hint": "full_object",
    },
    "n03970156": {
        "sense": "plunger cleaning tool with rubber suction cup and handle",
        "include_keywords": ["rubber cup", "wooden handle", "bathroom tool", "toilet plunger"],
        "exclude_keywords": ["syringe plunger", "industrial piston", "medical injection"],
        "scene_hint_override": "bathroom or utility room context with cleaning tools",
        "framing_hint": "close_up",
    },
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def _default_semantic_entry(rec: Dict) -> Dict:
    name = rec["name"]
    synonyms = rec.get("synonyms", [])
    syn_text = ", ".join(synonyms) if synonyms else name
    tokens = []
    for s in ([name] + synonyms):
        tokens.extend(_tokenize(s))
    uniq_tokens = []
    for t in tokens:
        if t not in uniq_tokens:
            uniq_tokens.append(t)
    include_keywords = uniq_tokens[:6] if uniq_tokens else _tokenize(name)

    return {
        "wnid": rec["wnid"],
        "name": name,
        "sense": f"photograph of {name} as the Tiny-ImageNet object class",
        "include_keywords": include_keywords[:8],
        "exclude_keywords": ["cartoon", "illustration", "icon", "logo"],
        "scene_hint_override": "",
        "framing_hint": "mid_shot",
        "synonyms": synonyms,
        "source_synonyms_text": syn_text,
    }


def build_semantic_spec(class_map: Dict) -> List[Dict]:
    rows = []
    for rec in class_map["classes"]:
        row = _default_semantic_entry(rec)
        hint = DISAMBIG_HINTS.get(rec["wnid"])
        if hint:
            row.update(hint)
        rows.append(row)
    return rows


def risk_score(row: Dict) -> Dict:
    score = 0
    reasons = []
    name = row["name"].lower()
    tokens = set(_tokenize(name))
    syn_cnt = len(row.get("synonyms", []))

    if len(name.replace(" ", "").replace("-", "")) <= 6:
        score += 2
        reasons.append("short_name<=6")
    if syn_cnt <= 1:
        score += 1
        reasons.append("synonyms<=1")
    if tokens.intersection(AMBIGUOUS_TOKENS):
        score += 3
        reasons.append("hits_ambiguous_token")
    if any(x in name for x in ["seal", "bass", "monitor", "trunk", "crane", "organ", "reel", "pole"]):
        score += 2
        reasons.append("known_polysemy")

    return {"risk_score": score, "risk_reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic spec and risk reports for Tiny-ImageNet classes.")
    parser.add_argument("--class-map", required=True, type=str)
    parser.add_argument("--semantic-spec-out", required=True, type=str)
    parser.add_argument("--risk-report-out", required=True, type=str)
    parser.add_argument("--risk-topk-out", required=True, type=str)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    rows = build_semantic_spec(class_map)

    # Coverage checks
    wnids = [r["wnid"] for r in rows]
    if len(rows) != 200:
        raise ValueError(f"Expected 200 classes, got {len(rows)}")
    if len(set(wnids)) != len(wnids):
        raise ValueError("Duplicate wnid found in semantic spec")

    spec_payload = {
        "version": "v1",
        "num_classes": len(rows),
        "classes": rows,
    }
    Path(args.semantic_spec_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.semantic_spec_out).write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = []
    for row in rows:
        rs = risk_score(row)
        report.append(
            {
                "wnid": row["wnid"],
                "name": row["name"],
                "synonym_count": len(row.get("synonyms", [])),
                **rs,
            }
        )
    report.sort(key=lambda x: (-x["risk_score"], x["name"]))
    topk = report[: args.topk]

    Path(args.risk_report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.risk_topk_out).write_text(json.dumps(topk, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] semantic spec: {args.semantic_spec_out}")
    print(f"[OK] risk report: {args.risk_report_out}")
    print(f"[OK] risk top{args.topk}: {args.risk_topk_out}")
    print(f"[OK] coverage: {len(rows)}/200, unique wnid={len(set(wnids))}")


if __name__ == "__main__":
    main()

