#!/usr/bin/env python3
"""
Regenerate run_tiny_imagenet_*_complete.sh from run_cifar10_* counterparts.
- Default attacks: same poison_rate block order as CIFAR (e.g. 0.05 -> 0.01 -> 0.005).
- SIG / UPGD: only poison_rate 0.005 and 0.001 (grouped: all 0.005 then all 0.001).
- Cross test: test_stl10.py -> test_tiny_imagenet.py + frost severity 2 and 3.
- Fixes stray CIFAR header lines (bare ===) into comments.
"""
from __future__ import annotations

import re
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def fix_broken_cifar_header(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("=" * 10) and not line.lstrip().startswith("#"):
            out.append("# " + line.strip())
        elif re.match(r"^Model:\s*\w+", s) and not line.lstrip().startswith(
            ("echo", "#")
        ):
            out.append("# " + line.strip())
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def apply_tiny_dataset_transform(text: str) -> str:
    text = text.replace("-dataset=cifar10", "-dataset=tiny_imagenet")
    text = text.replace("poisoned_train_set/cifar10/", "poisoned_train_set/tiny_imagenet/")
    for arch in ("mobilenetv2", "resnet18", "vgg19_bn"):
        text = text.replace(f"{arch}_cifar10", f"{arch}_tiny_imagenet")
    text = text.replace("CIFAR10", "TINY_IMAGENET")
    text = re.sub(r"\bcifar10\b", "tiny_imagenet", text)
    return text


def _split_run_command_double_quoted(line: str) -> tuple[str, str] | None:
    """Parse run_command \"CMD\" \"DESC\" (DESC ends with closing quote, no extra paren)."""
    prefix = 'run_command "'
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix) :]
    sep = '" "'
    i = rest.find(sep)
    if i < 0:
        return None
    cmd = rest[:i]
    desc_rest = rest[i + len(sep) :]
    if not desc_rest.endswith('"'):
        return None
    desc = desc_rest[:-1]
    return cmd, desc


def expand_cross_test_stl10_to_frost(text: str) -> str:
    """Each test_stl10.py run_command becomes two frost lines."""
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if (
            "test_stl10.py" in line
            and stripped.startswith("run_command")
            and "-corruption_type=frost" not in line
        ):
            parsed = _split_run_command_double_quoted(line)
            if not parsed:
                out_lines.append(line)
                continue
            cmd, desc = parsed
            cmd = cmd.replace("test_stl10.py", "test_tiny_imagenet.py")
            desc = desc.replace("test_stl10.py", "test_tiny_imagenet.py")
            for sev, sfx in ((2, "frost s=2"), (3, "frost s=3")):
                nc = cmd + f" -corruption_type=frost -severity={sev}"
                nd = desc + f" {sfx}"
                out_lines.append(f'run_command "{nc}" "{nd}"')
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _sig_upgd_flush_buffer(buf: list[str]) -> list[str]:
    if not buf:
        return []
    joined = "\n".join(buf)
    if "-poison_rate=0.05" not in joined and "-poison_rate=0.01" not in joined:
        return buf
    only_005 = [ln for ln in buf if "-poison_rate=0.005" in ln]

    def to001(ln: str) -> str:
        ln2 = re.sub(r"-poison_rate=0\.005\b", "-poison_rate=0.001", ln)
        ln2 = re.sub(r"rate=0\.005\b", "rate=0.001", ln2)
        return ln2

    return only_005 + [to001(ln) for ln in only_005]


def apply_sig_upgd_two_rates(text: str) -> str:
    """Between echo section markers, collapse CIFAR 0.05/0.01/0.005 to tiny 0.005 + 0.001."""
    lines = text.splitlines()
    out: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip().startswith("echo '-----") or line.strip().startswith(
            'echo "-----'
        ):
            out.extend(_sig_upgd_flush_buffer(buf))
            buf = []
            out.append(line)
        elif line.strip().startswith("run_command"):
            buf.append(line)
        else:
            out.extend(_sig_upgd_flush_buffer(buf))
            buf = []
            out.append(line)
    out.extend(_sig_upgd_flush_buffer(buf))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def parse_tiny_filename(name: str) -> tuple[str, str] | None:
    # run_tiny_imagenet_{attack}_{model}_complete.sh
    m = re.match(
        r"run_tiny_imagenet_([a-z0-9_]+)_(mobilenetv2|resnet18|vgg19_bn|vgg19)_complete\.sh$",
        name,
    )
    if not m:
        return None
    return m.group(1), m.group(2)


def cifar_source_path(attack: str, model: str) -> Path:
    return RUN_DIR / f"run_cifar10_{attack}_{model}_complete.sh"


def uncomment_basic_experiment_block(text: str) -> str:
    """CIFAR basic scripts may have the whole experiment commented out; enable for Tiny."""
    if "# echo '----- 1. Creation" not in text:
        return text
    if "\necho '----- 1. Creation" in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    started = False
    for line in lines:
        ls = line.lstrip()
        if "# echo '----- 1. Creation" in line:
            started = True
            pound = line.index("#")
            out.append(line[pound + 1 :].lstrip())
            continue
        if started:
            if ls.startswith("# echo ") or ls.startswith("# run_command"):
                pound = line.index("#")
                out.append(line[pound + 1 :].lstrip())
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def regenerate_one(tiny_path: Path, attack: str, model: str) -> None:
    cifar_path = cifar_source_path(attack, model)
    if not cifar_path.is_file():
        raise FileNotFoundError(f"Missing CIFAR reference: {cifar_path}")

    text = cifar_path.read_text(encoding="utf-8")
    text = fix_broken_cifar_header(text)

    if attack == "basic":
        text = uncomment_basic_experiment_block(text)

    if attack in ("sig", "upgd"):
        text = apply_sig_upgd_two_rates(text)

    text = apply_tiny_dataset_transform(text)
    text = expand_cross_test_stl10_to_frost(text)

    # Second echo line: attack name
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Attack Experiment Script" in line and "echo" in line:
            lines[i] = line.replace("CIFAR10", "TINY_IMAGENET")
            break
    text = "\n".join(lines) + "\n"

    tiny_path.write_text(text, encoding="utf-8")
    print(f"Wrote {tiny_path.name}")


def main() -> None:
    for p in sorted(RUN_DIR.glob("run_tiny_imagenet_*_complete.sh")):
        parsed = parse_tiny_filename(p.name)
        if not parsed:
            continue
        attack, model = parsed
        regenerate_one(p, attack, model)


if __name__ == "__main__":
    main()
