"""Step 1: read a CSV or JSON file and map whatever columns it has onto one standard shape.

We do not know what file the user will give us, so column names are detected, not assumed.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_NAMES = ["comment_text", "text_original", "text_display", "comment", "text", "message", "body", "content", "tweet_text", "tweet", "full_text",
              "review_text", "review", "post", "caption"]
ID_NAMES = ["comment_id", "id", "tweet_id", "post_id", "uid"]
PLATFORM_NAMES = ["platform", "source", "channel", "network", "site", "social_media"]
TIME_NAMES = ["created_at", "created_utc", "timestamp", "date", "datetime", "tweet_created", "posted_at", "published_at", "created_time", "time", "created"]
AUTHOR_NAMES = ["author", "author_display_name", "user", "username", "user_name", "screen_name", "handle", "name"]
LIKE_NAMES = ["likes", "like_count", "favorite_count", "favourite_count", "reactions", "upvotes", "ups", "score", "num_likes"]
SHARE_NAMES = ["shares", "share_count", "retweets", "retweet_count", "reposts", "num_shares"]
REPLY_NAMES = ["replies", "reply_count", "comments_count", "num_comments"]
EXACT_ONLY = {"id", "name", "score", "date", "time", "site", "ups", "source"}  # too general to match as the end of a longer name
MAX_TEXT = 5000  # one giant comment must not stall a run


class InputError(Exception):
    """The file cannot be used. The message is written for the user."""


def _decode(raw: bytes, path: Path) -> str:
    """Decode the whole file, because cutting bytes first can split a character and fool the check."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    text = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    sample = (text or "")[:4000]
    junk = sum((ord(c) < 32 and c not in "\t\r\n") or 127 <= ord(c) < 160 for c in sample)
    if text is None or junk > len(sample) * 0.02:
        raise InputError(f"{path} does not look like a text file.")
    return text


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise InputError(f"File not found: {path}")
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        return _read_json(path)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        return _read_csv(path)
    raise InputError(f"Unsupported file type '{path.suffix}'. Use .csv, .tsv, .json or .jsonl.")


def _read_csv(path: Path) -> pd.DataFrame:
    text = _decode(path.read_bytes(), path)
    if not text.strip():
        raise InputError(f"The file is empty: {path}")
    first_line = text.lstrip().splitlines()[0]
    sep = max("\t,;|", key=first_line.count) if any(c in first_line for c in "\t,;|") else ","
    csv.field_size_limit(10_000_000)
    try:
        rows = [r for r in csv.reader(io.StringIO(text, newline=""), delimiter=sep) if any(c.strip() for c in r)]
    except csv.Error as e:
        raise InputError(f"Could not read {path} as a table: {e}")
    header = _unique_names(rows[0])
    width = len(header)
    # Comments often hold unquoted commas, which split them into too many fields. Put the pieces back together.
    text_at = next((header.index(c) for n in TEXT_NAMES for c in header if _norm(c) == n), width - 1)
    records, repaired = [], 0
    for row in rows[1:]:
        if len(row) > width:
            extra = len(row) - width
            row = row[:text_at] + [sep.join(row[text_at:text_at + extra + 1])] + row[text_at + extra + 1:]
            repaired += 1
        records.append(row + [""] * (width - len(row)))
    df = pd.DataFrame(records, columns=header, dtype=str)
    df.attrs["repaired_rows"] = repaired
    return df


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        name = name.strip() or f"column_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _read_json(path: Path) -> pd.DataFrame:
    text = _decode(path.read_bytes(), path)
    if not text.strip():
        raise InputError(f"The file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:  # not one JSON document, so try one JSON object per line
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise InputError(f"{path} is not valid JSON or JSONL: {e}")
    records = _find_records(data)
    if not records:
        raise InputError(f"{path} has no list of comments inside it.")
    df = pd.json_normalize(records, sep="_")
    return df.astype(object).where(df.notna(), "").astype(str)


class _PlainStrings(list):
    """Records made from a plain list of strings, which may be comments or may only be tags."""


def _find_records(data, depth: int = 0) -> list[dict]:
    """Find the comments in any common JSON shape: a list, {"comments": [...]}, or something more deeply nested."""
    if isinstance(data, list):
        dicts = [x for x in data if isinstance(x, dict)]
        return dicts or _PlainStrings({"text": x} for x in data if isinstance(x, str))
    if isinstance(data, dict):
        if depth > 5:
            return []
        found = [f for f in (_find_records(v, depth + 1) for v in data.values() if isinstance(v, (list, dict))) if f]
        if found:
            # Prefer records that have a text field, then real records over plain strings, then the most text.
            return max(found, key=lambda f: (any(_norm(k) in TEXT_NAMES for k in f[0]) and not isinstance(f, _PlainStrings), not isinstance(f, _PlainStrings), sum(len(str(v)) for r in f for v in r.values())))
        return [data] if depth == 0 and any(isinstance(v, str) for v in data.values()) else []
    return []


def _norm(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))  # likeCount becomes like_count
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find(columns, names, exclude=(), accept=None) -> str | None:
    """Exact names first, then names at the end of a longer one, such as public_metrics_like_count."""
    normed = [(_norm(c), c) for c in columns if c not in exclude]
    for n in names:
        for key, col in normed:
            if key == n and (accept is None or accept(col)):
                return col
    for n in names:
        if n in EXACT_ONLY:
            continue
        for key, col in normed:
            if key.endswith("_" + n) and (accept is None or accept(col)):
                return col
    return None


def _guess_text_column(df: pd.DataFrame, exclude=()) -> str | None:
    """No familiar name, so pick the column that looks most like human writing."""
    best, best_score = None, 0.0
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].astype(str).head(2000)
        has_words = s.str.contains(r"[^\W\d_]{2,}\s+[^\W\d_]{2,}", regex=True).mean()
        score = s.str.len().mean() * s.nunique() / max(len(s), 1) * has_words
        if score > best_score:
            best, best_score = col, score
    return best if best_score >= 5 else None


def _to_number(s: pd.Series) -> pd.Series:
    """Accept 1,200 and 1.2K. Anything else counts as 0."""
    t = s.astype(str).str.strip().str.replace(",", "", regex=False).str.lower()
    short = t.str.extract(r"^([\d.]+)\s*([km])$")
    scaled = pd.to_numeric(short[0], errors="coerce") * short[1].map({"k": 1e3, "m": 1e6})
    number = pd.to_numeric(t, errors="coerce").fillna(scaled)
    return number.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)


def parse_times(s: pd.Series) -> pd.Series:
    """Accept epoch numbers and ISO dates. For ambiguous dates, try day-first and month-first and keep the better fit."""
    text = s.astype(str).str.strip().replace({"": None, "nan": None, "None": None})
    numbers = pd.to_numeric(text, errors="coerce")
    if numbers.notna().sum() > 0.8 * text.notna().sum() and numbers.median() > 1e8:
        parsed = pd.to_datetime(numbers, errors="coerce", unit="ms" if numbers.median() > 1e11 else "s", utc=True)
    else:
        parsed = pd.to_datetime(text, errors="coerce", format="ISO8601", utc=True)
        rest = parsed.isna() & text.notna()
        if rest.any():
            day_first = pd.to_datetime(text[rest], errors="coerce", format="mixed", dayfirst=True, utc=True)
            month_first = pd.to_datetime(text[rest], errors="coerce", format="mixed", dayfirst=False, utc=True)
            span = lambda d: (d.max() - d.min()) if d.notna().any() else pd.Timedelta.max
            better = day_first if (day_first.notna().sum(), -span(day_first).total_seconds()) >= (month_first.notna().sum(), -span(month_first).total_seconds()) else month_first
            parsed.loc[rest] = better
    parsed = parsed.dt.tz_localize(None)
    # A few wildly wrong dates, such as 1970-01-01, must not stretch the whole timeline.
    far = (parsed - parsed.median()).abs() > pd.Timedelta(days=365)
    if 0 < far.mean() < 0.01:
        parsed = parsed.mask(far)
    return parsed


def load_comments(path: str | Path, text_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a table with the columns id, text, platform, created_at, author, engagement, and a note of what was found."""
    raw = read_table(path)
    if raw.empty:
        raise InputError(f"No rows found in {path}.")
    if text_col and text_col not in raw.columns:
        raise InputError(f"Column '{text_col}' is not in the file. Columns found: {list(raw.columns)}")

    def looks_like_text(col) -> bool:
        return raw[col].astype(str).head(2000).str.contains(r"[^\W\d_]{2}", regex=True).mean() >= 0.3

    id_col = _find(raw.columns, ID_NAMES)
    text_col = text_col or _find(raw.columns, TEXT_NAMES, exclude=(id_col,), accept=looks_like_text) or _guess_text_column(raw, exclude=(id_col,))
    if text_col is None:
        raise InputError(f"Could not find a comment text column. Columns found: {list(raw.columns)}. Use --text-col to name it.")

    used = {text_col, id_col}
    platform_col = _find(raw.columns, PLATFORM_NAMES, exclude=used)
    time_col = _find(raw.columns, TIME_NAMES, exclude=used)
    author_col = _find(raw.columns, AUTHOR_NAMES, exclude=used)
    like_col = _find(raw.columns, LIKE_NAMES, exclude=used)
    share_col = _find(raw.columns, SHARE_NAMES, exclude=used)
    reply_col = _find(raw.columns, REPLY_NAMES, exclude=used)

    df = pd.DataFrame(index=raw.index)
    text = raw[text_col].astype(str).replace({"nan": ""})
    df["text"] = text.str[:MAX_TEXT]

    # Ids must be unique because every claim in the report is traced back to them. Repeats get a suffix, so they still join to the input.
    generated = pd.Series([f"c{i + 1}" for i in range(len(raw))], index=raw.index)
    ids = raw[id_col].astype(str).str.strip() if id_col else generated
    ids = ids.where(ids != "", generated)
    repeat = ids.groupby(ids).cumcount()
    df["id"] = ids.where(repeat == 0, ids + "#" + (repeat + 1).astype(str))

    platform = raw[platform_col].astype(str).str.strip() if platform_col else pd.Series("", index=raw.index)
    df["platform"] = platform.replace({"": "unknown", "nan": "unknown"})
    df["created_at"] = parse_times(raw[time_col]) if time_col else pd.NaT
    df["author"] = raw[author_col].astype(str) if author_col else ""
    likes = _to_number(raw[like_col]) if like_col else 0
    shares = _to_number(raw[share_col]) if share_col else 0
    replies = _to_number(raw[reply_col]) if reply_col else 0
    df["engagement"] = (likes + shares + replies) if (like_col or share_col or reply_col) else 0
    df["engagement"] = np.asarray(df["engagement"], dtype=float)

    info = {"rows": int(len(df)), "text_col": text_col, "id_col": id_col, "platform_col": platform_col, "time_col": time_col,
            "author_col": author_col, "engagement_cols": [c for c in (like_col, share_col, reply_col) if c],
            "repaired_rows": int(raw.attrs.get("repaired_rows", 0)), "duplicate_ids": int((repeat > 0).sum()), "cut_long_comments": int((text.str.len() > MAX_TEXT).sum()),
            "dates_parsed": round(float(df["created_at"].notna().mean()), 3) if time_col else None}
    return df.reset_index(drop=True), info
