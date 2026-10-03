"""Output schemas: normalization (JSON Schema dict or pydantic model), strict/lenient parsing, validation.

Parsing is strict first (the whole answer, after stripping a reasoning block and a code fence, must be
one JSON value). The lenient extractor (first balanced JSON object anywhere in the text) is a
*counted* fallback: results recovered that way carry ``lenient_parse=True``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_FENCE_ANY = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.DOTALL)
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def is_pydantic_model(schema: Any) -> bool:
    return isinstance(schema, type) and hasattr(schema, "model_json_schema") and hasattr(schema, "model_validate")


def schema_to_dict(schema: Any) -> Optional[Dict[str, Any]]:
    """JSON Schema dict for ``schema`` (``None`` for free-text requests)."""
    if schema is None:
        return None
    if is_pydantic_model(schema):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise TypeError(f"output_schema must be a JSON Schema dict or a pydantic model class, got {type(schema)!r}")


def schema_name(schema: Any) -> str:
    if is_pydantic_model(schema):
        return re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64]
    if isinstance(schema, dict) and isinstance(schema.get("title"), str):
        return re.sub(r"[^A-Za-z0-9_-]", "_", schema["title"])[:64] or "judge_output"
    return "judge_output"


def split_reasoning(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a leading ``<think>...</think>`` block from the answer. Returns ``(answer, reasoning)``.

    An unterminated ``<think>`` (generation stopped mid-reasoning) yields an empty answer.
    """
    if text is None:
        return None, None
    m = _THINK.match(text)
    if m:
        return text[m.end():], m.group(1).strip()
    stripped = text.lstrip()
    if stripped.startswith("<think>") and "</think>" not in stripped:
        return "", stripped[len("<think>"):].strip()
    return text, None


def parse_strict(text: str) -> Tuple[bool, Any]:
    """Whole-answer JSON parse (a single surrounding code fence is allowed)."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return True, json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return False, None


def extract_json(text: str) -> Optional[Any]:
    """Lenient: first fenced JSON object, else the first balanced ``{...}`` that parses."""
    if not text:
        return None
    for m in _FENCE_ANY.finditer(text):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ----------------------------------------------------------------------------------------- validation
def validate(schema: Any, value: Any) -> Tuple[Any, List[str]]:
    """Validate ``value``. Returns ``(parsed, errors)``; for a pydantic model ``parsed`` is an instance."""
    if schema is None:
        return value, []
    if is_pydantic_model(schema):
        try:
            return schema.model_validate(value), []
        except Exception as e:  # pydantic.ValidationError
            errs = getattr(e, "errors", None)
            if callable(errs):
                return None, [f"{'.'.join(str(p) for p in er.get('loc', ())) or '<root>'}: {er.get('msg')}" for er in errs()][:10]
            return None, [str(e)[:500]]
    return value, validate_json_schema(schema, value)


def validate_json_schema(schema: Dict[str, Any], value: Any) -> List[str]:
    """Errors of ``value`` against a JSON Schema (uses ``jsonschema`` when installed)."""
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover - exercised via the mini validator test
        return mini_validate(schema, value)
    cls = jsonschema.validators.validator_for(schema)
    try:
        v = cls(schema)
    except Exception as e:  # invalid schema is a caller bug, surface it
        return [f"invalid schema: {e}"]
    out = []
    for err in sorted(v.iter_errors(value), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        out.append(f"{loc}: {err.message}"[:300])
        if len(out) >= 10:
            break
    return out


_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _type_ok(t: str, value: Any) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(t)
    return True if py is None else isinstance(value, py)


def mini_validate(schema: Dict[str, Any], value: Any, path: str = "<root>") -> List[str]:
    """Dependency-free subset of JSON Schema: type, enum, const, properties, required,
    additionalProperties (bool), items, min/maxItems, minimum/maximum, min/maxLength, anyOf."""
    errs: List[str] = []
    if not isinstance(schema, dict):
        return errs
    if "anyOf" in schema:
        if not any(not mini_validate(s, value, path) for s in schema["anyOf"]):
            errs.append(f"{path}: does not match any allowed alternative")
        return errs
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_type_ok(x, value) for x in ts):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing required property {k!r}")
        for k, v in value.items():
            if k in props:
                errs.extend(mini_validate(props[k], v, f"{k}" if path == "<root>" else f"{path}.{k}"))
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property {k!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs.extend(mini_validate(schema["items"], v, f"{path}[{i}]"))
    return errs[:10]


def schema_instruction(schema_dict: Dict[str, Any]) -> str:
    """Instruction appended when the backend cannot enforce the schema natively."""
    return ("\n\nRespond with a single JSON object that conforms to this JSON Schema, and nothing else:\n"
            + json.dumps(schema_dict, ensure_ascii=False, sort_keys=True))


def with_schema_instruction(messages: List[Dict[str, Any]], schema_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append the schema instruction to the last user message (text content or content parts)."""
    out = [dict(m) for m in messages]
    instr = schema_instruction(schema_dict)
    for m in reversed(out):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = c + instr
            elif isinstance(c, list):
                m["content"] = list(c) + [{"type": "text", "text": instr}]
            return out
    out.append({"role": "user", "content": instr.strip()})
    return out


def correction_message(errors: List[str]) -> str:
    return ("Your previous answer was rejected: " + "; ".join(errors[:5])
            + ". Reply again with only a corrected JSON object.")
