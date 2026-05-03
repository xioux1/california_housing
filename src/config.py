from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(v) for v in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            new_dict: dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping/dictionary.")
    return data


def _dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_dump_yaml(v, indent + 2))
        elif isinstance(v, list):
            vals = ", ".join(str(x) for x in v)
            lines.append(f"{pad}{k}: [{vals}]")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def save_config(config: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_dump_yaml(config) + "\n", encoding="utf-8")


def save_json(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
