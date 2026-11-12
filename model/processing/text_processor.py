"""
Text Processor and Normalizer for UMKM Customer Reviews.
Cleans slang, normalizes Indonesian/Surabayan terms, and detects digital readiness keywords.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from src.umkm.processing.anonymizer import TextAnonymizer


class TextProcessor:
    """Cleans, normalizes, and extracts indicator signals from text reviews."""

    # Common Indonesian and Surabayan culinary slang mapping
    SLANG_DICT = {
        "bgt": "banget",
        "bgtt": "banget",
        "gak": "tidak",
        "ga": "tidak",
        "ngga": "tidak",
        "nggak": "tidak",
        "ndak": "tidak",
        "dgn": "dengan",
        "dg": "dengan",
        "utk": "untuk",
        "klo": "kalau",
        "kalo": "kalau",
        "tp": "tapi",
        "tpi": "tapi",
        "sdh": "sudah",
        "udah": "sudah",
        "udh": "sudah",
        "blm": "belum",
        "bener": "benar",
        "bgt": "banget",
        "mantep": "mantap",
        "enak bgt": "enak sekali",
        "recomended": "direkomendasikan",
        "recommended": "direkomendasikan",
        "rekomen": "direkomendasikan",
        "pake": "pakai",
        "pakek": "pakai",
        "pesen": "pesan",
        "byr": "bayar",
        "byar": "bayar",
        "gofood": "gofood",
        "go food": "gofood",
        "grabfood": "grabfood",
        "grab food": "grabfood",
        "shopeefood": "shopeefood",
        "shopee food": "shopeefood",
        "ojol": "ojek_online",
        "qris": "qris",
        "barkod": "qris",
        "barcode": "qris",
        "scan": "qris",
        "debit": "edc",
        "kartu": "edc",
        "edc": "edc",
        "mesin edc": "edc",
        "kasir": "pos",
        "moka": "pos",
        "pawoon": "pos",
        "tunai": "tunai",
        "cash": "tunai",
    }

    DIGITAL_KEYWORDS = {
        "qris": ["qris", "barcode", "scan", "gopay", "ovo", "dana", "shopeepay"],
        "pos": ["kasir", "mesin kasir", "struk", "pos", "moka", "pawoon"],
        "edc": ["edc", "debit", "kartu kredit", "gesek", "bca", "mandiri", "bri"],
        "merchant_online": ["gofood", "grabfood", "shopeefood", "ojol", "pesan antar", "delivery"],
    }

    @classmethod
    def clean_and_normalize(cls, raw_text: str) -> Dict[str, Any]:
        """Anonymize, normalize slang, and extract keyword indicators from text."""
        # 1. Anonymize PII
        anonymized = TextAnonymizer.anonymize_text(raw_text)

        # 2. Lowercase & strip special symbols
        clean_text = anonymized.lower()
        clean_text = re.sub(r"[^\w\s]", " ", clean_text)
        words = clean_text.split()

        # 3. Slang replacement
        normalized_words = [cls.SLANG_DICT.get(w, w) for w in words]
        normalized_sentence = " ".join(normalized_words)

        # 4. Keyword indicator detection in review
        signals = {
            "mention_qris": any(k in normalized_sentence for k in cls.DIGITAL_KEYWORDS["qris"]),
            "mention_pos": any(k in normalized_sentence for k in cls.DIGITAL_KEYWORDS["pos"]),
            "mention_edc": any(k in normalized_sentence for k in cls.DIGITAL_KEYWORDS["edc"]),
            "mention_online": any(k in normalized_sentence for k in cls.DIGITAL_KEYWORDS["merchant_online"]),
        }

        return {
            "teks_anonim": anonymized,
            "teks_bersih": normalized_sentence,
            "word_count": len(normalized_words),
            **signals,
        }
