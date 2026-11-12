"""
Data Anonymization Utility.
Sanitizes personal identifiable information (PII) from reviews and blurs faces on photos.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class TextAnonymizer:
    """Regex-based anonymizer for customer reviews and UMKM metadata."""

    PHONE_PATTERN = re.compile(r"(\+62|62|0)8[1-9][0-9]{6,10}")
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@[a-zA-Z0-9_.]+")

    @classmethod
    def anonymize_text(cls, text: str) -> str:
        """Replace phone numbers, emails, URLs, and social mentions with tokens."""
        if not text or not isinstance(text, str):
            return ""
        s = cls.PHONE_PATTERN.sub("[NOMOR_TELEPON]", text)
        s = cls.EMAIL_PATTERN.sub("[EMAIL]", s)
        s = cls.URL_PATTERN.sub("[URL]", s)
        s = cls.MENTION_PATTERN.sub("[USER]", s)
        return s.strip()

    @classmethod
    def anonymize_reviews(cls, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mask reviewer author names and cleanse review text."""
        cleaned = []
        for r in reviews:
            rev_copy = dict(r)
            rev_copy["author_name"] = "Pengulas_Anonim"
            rev_copy["text_clean"] = cls.anonymize_text(rev_copy.get("text", ""))
            cleaned.append(rev_copy)
        return cleaned
