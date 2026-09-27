"""Transcript language screen for demo clips (whisper words in the worker ``<vid>.json``).

Two tiers: ``strict`` (profanity, sexual/anatomical terms, slurs, any token with ``*`` = a censored word) and
``mild`` (minced oaths such as "freaking", "damn", "hell"). A transcript cannot see bleeped audio, music lyrics or
on-screen text, so a clean screen is necessary, not sufficient: someone still watches the finalists once.
"""

from __future__ import annotations

import re

STRICT_STEMS = ("fuck", "shit", "bitch", "cunt", "nigg", "fagg", "retard", "masturbat", "porn", "dildo", "orgasm",
                "whore", "slut", "motherf", "bullshit", "dumbass", "jackass", "asshole", "dickhead", "penis",
                "vagin", "erectil", "boob", "titt", "blowjob", "handjob")
STRICT_EXACT = {"ass", "asses", "dick", "dicks", "cock", "cocks", "pussy", "pussies", "tits", "sex", "sexy",
                "sexual", "sexually", "cum", "horny", "fag", "fags", "bastard", "bastards", "nude", "nudes",
                "naked", "prick", "twat", "wank", "wanker", "hoe", "hoes", "nsfw", "stfu", "wtf", "af"}
MILD_EXACT = {"damn", "dammit", "damnit", "goddamn", "goddamnit", "hell", "crap", "crappy", "freaking", "freakin",
              "frickin", "fricking", "friggin", "frigging", "effing", "piss", "pissed", "pissing", "bloody", "sucks"}

_SUFFIX = re.compile(r"'(s|re|ll|ve|d|m)$")
_NONWORD = re.compile(r"[^a-z*]+")


def normalise(token: str) -> str:
    """Lower case, contraction suffix dropped ("you're" -> "you"), letters and '*' only."""
    return _NONWORD.sub("", _SUFFIX.sub("", token.lower().replace("’", "'").strip(".,!?;:\"()")))


def tier(token: str) -> str | None:
    """'strict', 'mild' or None for one transcript token."""
    t = normalise(token)
    if not t:
        return None
    if "*" in t:
        return "strict"
    if t in STRICT_EXACT or any(t.startswith(s) or s in t for s in STRICT_STEMS):
        return "strict"
    if t in MILD_EXACT:
        return "mild"
    return None


def screen(words: list[dict]) -> dict:
    """Counts per tier over a whole transcript (list of {'text': ...}); the flagged words are not returned."""
    n = {"strict": 0, "mild": 0}
    for w in words or []:
        k = tier(str(w.get("text", "")))
        if k:
            n[k] += 1
    return {"n_strict": n["strict"], "n_mild": n["mild"], "n_words": len(words or []),
            "clean": n["strict"] == 0 and n["mild"] == 0}
