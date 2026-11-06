"""Shared loading and analysis functions for the local Signal Desk app."""
from __future__ import annotations

import html
import io
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

TEXT_NAMES = ("text", "tweet", "content", "message", "post", "original_text")
DATE_NAMES = ("date", "timestamp", "created_at", "time", "datetime")
USER_NAMES = ("user", "username", "author", "handle", "screen_name")
TRANSLATION_NAMES = ("translation", "translated_text", "translation_text")

DEMO = [
    {"user": "City updates", "date": "2026-11-02 09:14", "text": "Heavy rain is causing flooding near the river walk. Please avoid the area and check local updates. #weather #safety"},
    {"user": "Maya Chen", "date": "2026-11-02 08:56", "text": "Such a lovely morning for a walk. Coffee, sunshine, and a new playlist. #weekend"},
    {"user": "Jordan Lee", "date": "2026-11-02 08:27", "text": "Anyone else having trouble with the app after the latest update? It keeps crashing when I upload photos. #iPhone"},
    {"user": "Tech Brief", "date": "2026-11-01 21:34", "text": "The latest phone launch has people talking about battery life and new camera features. #iPhone #tech"},
    {"user": "Community Watch", "date": "2026-11-01 19:58", "text": "Power is out on several streets after the storm. Crews are responding; report downed lines to the utility. #storm"},
    {"user": "Sam Rivera", "date": "2026-11-01 18:16", "text": "Disappointed with the delivery delay. Support was helpful, though, and sorted it out quickly."},
    {"user": "Nina Patel", "date": "2026-10-31 17:41", "text": "Finally tried the new camera in low light and wow, these photos came out beautifully. #photography #iPhone"},
    {"user": "Harbour News", "date": "2026-10-31 16:02", "text": "Emergency crews are assisting residents after reports of rising water near the waterfront. #flood"},
    {"user": "Alex Morgan", "date": "2026-10-30 13:18", "text": "Not impressed by the new subscription price. The features are nice but this is too expensive."},
    {"user": "Luca", "date": "2026-10-29 20:05", "text": "Tried the new recipe everyone is sharing. Easy to make and genuinely delicious! #food"},
]

POSITIVE = set("amazing awesome beautiful best better brilliant delightful easy excellent excited fantastic favorite good gorgeous great happy helpful incredible love lovely nice perfect pleased quick recommend smooth solved stunning thank thanks wonderful wow".split())
NEGATIVE = set("angry annoying awful bad broken cancel crash crashing disappointing disappointed expensive fail failed frustrating hate issue late lost problem poor terrible trouble unfortunately unhappy worst".split())

CRISIS_TERMS = {
    "en": {"flood": ["flood", "flooding", "flash flood", "rising water"], "fire": ["wildfire", "bushfire", "fire", "smoke"], "earthquake": ["earthquake", "aftershock"], "storm": ["cyclone", "hurricane", "typhoon", "storm", "heavy rain"], "medical": ["outbreak", "epidemic", "pandemic"]},
    "es": {"flood": ["inundación", "inundaciones", "crecida"], "fire": ["incendio", "incendios", "fuego"], "earthquake": ["terremoto", "sismo"], "storm": ["huracán", "ciclón", "tormenta"], "medical": ["brote", "epidemia"]},
    "fr": {"flood": ["inondation", "inondations", "crue"], "fire": ["incendie", "feu de forêt"], "earthquake": ["séisme", "tremblement de terre"], "storm": ["ouragan", "cyclone", "tempête"], "medical": ["épidémie", "flambée"]},
    "de": {"flood": ["überschwemmung", "hochwasser"], "fire": ["waldbrand", "brand"], "earthquake": ["erdbeben"], "storm": ["orkan", "sturm"], "medical": ["ausbruch", "epidemie"]},
    "pt": {"flood": ["inundação", "enchente"], "fire": ["incêndio", "queimada"], "earthquake": ["terremoto", "sismo"], "storm": ["furacão", "tempestade"], "medical": ["surto", "epidemia"]},
}
URGENCY = {
    "en": ["urgent", "emergency", "help", "trapped", "evacuate", "missing", "injured", "sos", "rescue", "stranded"],
    "es": ["urgente", "emergencia", "ayuda", "atrapado", "evacuar", "desaparecido", "herido"],
    "fr": ["urgence", "urgent", "aidez", "secours", "bloqué", "évacuer", "blessé"],
    "de": ["notfall", "dringend", "hilfe", "eingeschlossen", "evakuieren", "vermisst", "verletzt"],
    "pt": ["urgente", "emergência", "socorro", "ajuda", "preso", "evacuar", "ferido"],
}


def preprocess(text: object) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w]+", " ", text)
    text = re.sub(r"#([\w]+)", r"\1", text)
    text = text.lower()
    for a, b in ((r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"), (r"'m\b", " am"), (r"'s\b", " is")):
        text = re.sub(a, b, text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)).strip()


def _col(columns, names):
    normalized = {re.sub(r"[ _-]+", "", str(c).lower()): c for c in columns}
    for name in names:
        key = re.sub(r"[ _-]+", "", name.lower())
        if key in normalized:
            return normalized[key]
    return None


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    text_col = _col(frame.columns, TEXT_NAMES)
    if text_col is None and len(frame.columns) >= 6:
        # Headerless Sentiment140: target, id, date, flag, user, text.
        frame = frame.rename(columns={frame.columns[0]: "target", frame.columns[1]: "id", frame.columns[2]: "date", frame.columns[4]: "user", frame.columns[5]: "text"})
        text_col = "text"
    if text_col is None:
        raise ValueError("Could not find post text. Add a text, tweet, content, or message column.")
    date_col, user_col = _col(frame.columns, DATE_NAMES), _col(frame.columns, USER_NAMES)
    out = pd.DataFrame({"text": frame[text_col].fillna("").astype(str)})
    out["date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    out["user"] = frame[user_col].fillna("Unknown").astype(str) if user_col else "Unknown"
    translation_col = _col(frame.columns, TRANSLATION_NAMES)
    if translation_col:
        out["translation"] = frame[translation_col].fillna("").astype(str)
    id_col = _col(frame.columns, ("id", "post_id"))
    out["id"] = frame[id_col].astype(str) if id_col else np.arange(len(out)).astype(str)
    target_col = _col(frame.columns, ("target", "sentiment", "label"))
    out["target"] = pd.to_numeric(frame[target_col], errors="coerce") if target_col else np.nan
    # Sentiment140 labels map to binary 0/1; already binary labels stay as supplied.
    labels = set(out["target"].dropna().unique())
    if 4 in labels:
        out["target"] = out["target"].map({0: 0, 4: 1})
    out = out[out.text.str.strip().ne("")].drop_duplicates(subset=["text"]).reset_index(drop=True)
    out["processed_text"] = out.text.map(preprocess)
    return out


def read_upload(name: str, raw: bytes) -> pd.DataFrame:
    if name.lower().endswith(".json") or name.lower().endswith(".jsonl"):
        if name.lower().endswith(".jsonl"):
            frame = pd.read_json(io.BytesIO(raw), lines=True)
        else:
            obj = json.loads(raw.decode("utf-8-sig"))
            if isinstance(obj, dict):
                obj = next((obj[k] for k in ("data", "posts", "tweets") if isinstance(obj.get(k), list)), [obj])
            frame = pd.DataFrame(obj if isinstance(obj, list) else [obj])
    else:
        # Sentiment140 is Latin-1; UTF-8 is preferred for ordinary exports.
        try:
            frame = pd.read_csv(io.BytesIO(raw))
            if _col(frame.columns, TEXT_NAMES) is None:
                frame = pd.read_csv(io.BytesIO(raw), header=None, encoding="latin-1")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(raw), header=None, encoding="latin-1")
    return normalize_frame(frame)


def sentiment(text: str) -> tuple[str, int]:
    words = re.findall(r"[a-z']+", str(text).lower())
    value = sum((w in POSITIVE) - (w in NEGATIVE) for w in words)
    return ("positive" if value > 0 else "negative" if value < 0 else "neutral", value)


def hashtag_counts(df: pd.DataFrame) -> Counter:
    counts = Counter()
    for text in df.text:
        counts.update(set(t.lower() for t in re.findall(r"(?<!\w)#([\w]+)", str(text), re.UNICODE)))
    return counts


def mention_counts(df: pd.DataFrame) -> Counter:
    return Counter(m.lower() for text in df.text for m in re.findall(r"(?<!\w)@([\w]+)", str(text), re.UNICODE))


LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese", "zh-tw": "Chinese",
}


def _script_profile(text: str) -> tuple[str, bool]:
    counts = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        script = next((s for s in ("LATIN", "CYRILLIC", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "CJK", "HANGUL", "DEVANAGARI", "THAI") if s in name), "OTHER")
        counts[script] += 1
    if not counts:
        return "UNKNOWN", False
    script = counts.most_common(1)[0][0]
    mixed = sum(count >= 3 for count in counts.values()) > 1
    return script, mixed


def normalize_multilingual_text(text: object, language: str = "") -> str:
    """Normalize social text while preserving the original for side-by-side review."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"#([\w]+)", r"\1", value)
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    slang = {
        "en": {"pls": "please", "u": "you"},
        "es": {"xfa": "por favor", "ayudaaa": "ayuda"},
    }
    for short, expanded in slang.get(language, {}).items():
        value = re.sub(rf"(?<!\w){re.escape(short)}(?!\w)", expanded, value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def detect_language(text: object) -> dict:
    """Detect language/script for review; returns method and uncertainty explicitly."""
    original = str(text or "")
    script, mixed = _script_profile(original)
    code, confidence, method = "und", 0.0, "fallback"
    if len(original.strip()) >= 15:
        try:
            from langdetect import DetectorFactory, detect_langs
            DetectorFactory.seed = 0
            predictions = detect_langs(original)
            if predictions:
                code, confidence, method = predictions[0].lang.lower(), float(predictions[0].prob), "langdetect"
        except ImportError:
            pass
        except Exception:
            pass
    if code == "und":
        script_fallback = {"LATIN": "en", "ARABIC": "ar", "CYRILLIC": "ru", "HIRAGANA": "ja", "KATAKANA": "ja", "CJK": "zh-cn", "HANGUL": "ko", "THAI": "th"}
        code = script_fallback.get(script, "und")
        confidence = 0.25 if code != "und" else 0.0
        method = "script fallback" if code != "und" else "unknown"
    return {
        "language_code": code,
        "language": LANGUAGE_NAMES.get(code, LANGUAGE_NAMES.get(code.split("-")[0], "Undetermined")),
        "confidence": round(confidence, 3),
        "script": script,
        "mixed_script": mixed,
        "method": method,
        "normalized_text": normalize_multilingual_text(original, code.split("-")[0]),
    }


def multilingual_analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze language and script only; crisis classification is a separate step."""
    rows = []
    for _, post in df.iterrows():
        detected = detect_language(post.text)
        row = {
            "Post ID": post.id,
            "Date": post.date,
            "User": post.user,
            "Original text": post.text,
            "Language": detected["language"],
            "Language code": detected["language_code"],
            "Confidence": detected["confidence"],
            "Detection method": detected["method"],
            "Script": detected["script"],
            "Mixed script": detected["mixed_script"],
            "Normalized text": detected["normalized_text"],
        }
        if "translation" in df.columns:
            row["Supplied translation"] = post.get("translation", "")
        rows.append(row)
    columns = ["Post ID", "Date", "User", "Original text", "Language", "Language code", "Confidence", "Detection method", "Script", "Mixed script", "Normalized text"]
    if "translation" in df.columns:
        columns.append("Supplied translation")
    return pd.DataFrame(rows, columns=columns)


def crisis_analyze(text: str) -> dict:
    folded = str(text).casefold()
    languages = [("en", 0)]
    # Tiny language cue list makes the baseline transparent; it is not language ID.
    language_words = {"es": ["inundación", "incendio", "terremoto", "huracán", "ayuda"], "fr": ["inondation", "incendie", "séisme", "ouragan", "secours"], "de": ["überschwemmung", "waldbrand", "erdbeben", "orkan", "hilfe"], "pt": ["inundação", "incêndio", "terremoto", "furacão", "socorro"]}
    matches_lang = [(lang, sum(w in folded for w in words)) for lang, words in language_words.items()]
    best = max(matches_lang, key=lambda x: x[1]) if matches_lang else ("en", 0)
    lang = best[0] if best[1] else "en"
    matches = [(kind, term) for kind, terms in CRISIS_TERMS[lang].items() for term in terms if term.casefold() in folded]
    urgencies = [term for term in URGENCY[lang] if term.casefold() in folded]
    score = min(1.0, .25 * len(urgencies) + (.35 if "sos" in folded else 0))
    return {"language": lang, "crisis_type": ", ".join(sorted(set(k for k, _ in matches))) or "unclassified", "crisis_evidence": ", ".join(sorted(set(t for _, t in matches))), "urgency": "high" if score >= .7 else "medium" if score >= .3 else "unconfirmed", "urgency_score": score, "urgency_evidence": ", ".join(urgencies)}


def demo_frame() -> pd.DataFrame:
    return normalize_frame(pd.DataFrame(DEMO))


def latest_date(df: pd.DataFrame):
    valid = df.date.dropna()
    return valid.max() if len(valid) else None
