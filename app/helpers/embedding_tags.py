"""Small, Qt-free helpers for embedding tags.

Kept separate so the UI (dialog / button), the search worker thread and the
save/load code all share exactly the same rules.
"""

import re
from typing import Iterable, List, Sequence, Union

MAX_TAG_LENGTH = 64

_SPLIT_RE = re.compile(r"[,;\n\r]+")
_SPACES_RE = re.compile(r"\s+")


def normalize_tags(value: Union[str, Iterable[str], None]) -> List[str]:
    """Turn user input (or a stored list) into a clean list of tags.

    - a string is split on commas / semicolons / new lines
    - whitespace is trimmed and collapsed, a leading '#' is dropped
    - empty tags are removed, duplicates are removed case-insensitively
      (the first spelling wins), each tag is capped at MAX_TAG_LENGTH chars
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = _SPLIT_RE.split(value)
    else:
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.extend(_SPLIT_RE.split(item))

    result: List[str] = []
    seen = set()
    for part in parts:
        tag = _SPACES_RE.sub(" ", part).strip().lstrip("#").strip()
        tag = tag[:MAX_TAG_LENGTH].strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


def tags_match(search_text: str, name: str, tags_lower: Sequence[str]) -> bool:
    """Embedding search rule.

    `search_text` must already be lower-case; `tags_lower` are the embedding's
    tags in lower-case.  A plain search matches the name OR any tag (substring).
    A search starting with '#' only looks at tags ('#' alone matches everything).
    """
    if not search_text:
        return True
    if search_text.startswith("#"):
        term = search_text[1:].strip()
        if not term:
            return True
        return any(term in tag for tag in tags_lower)
    if search_text in name.lower():
        return True
    return any(search_text in tag for tag in tags_lower)
