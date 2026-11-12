"""
Text Processor and Normalizer for UMKM Customer Reviews.
Preserves original default language (Indonesian, English, Javanese) without forced translation.
Cleans slang only for colloquial Indonesian/Surabayan terms while maintaining text integrity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from model.processing.anonymizer import TextAnonymizer


class TextProcessor:
    """Cleans, normalizes, and extracts indicator signals from text reviews in their default language."""

    # Common Indonesian and Surabayan colloquial slang mapping (slang -> standard Indonesian)
    # Strictly abbreviations and colloquialisms; NO cross-language replacements (e.g. recommended/cash remain untouched).
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
        "mantep": "mantap",
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
        "kartu": "kartu",
        "edc": "edc",
        "kasir": "pos",
        "moka": "pos",
        "pawoon": "pos",
    }

    # Multilingual digital readiness indicator keywords (supports ID, EN, JV natively)
    DIGITAL_KEYWORDS = {
        "qris": [
            "qris", "qr code", "qr", "barcode", "scan", "cashless", "gopay",
            "ovo", "dana", "shopeepay", "e-wallet", "ewallet", "digital payment"
        ],
        "pos": [
            "kasir", "mesin kasir", "struk", "pos", "moka", "pawoon",
            "cashier", "receipt", "point of sale", "cash register"
        ],
        "edc": [
            "edc", "debit", "kartu kredit", "credit card", "card payment",
            "gesek", "bca", "mandiri", "bri", "card machine"
        ],
        "merchant_online": [
            "gofood", "grabfood", "shopeefood", "ojol", "pesan antar",
            "delivery", "online delivery", "food delivery", "take away", "takeaway"
        ],
    }

    EN_INDICATORS = {
        "the", "and", "is", "was", "are", "were", "this", "that", "with", "for",
        "from", "food", "place", "very", "good", "delicious", "service", "ordered",
        "order", "visit", "visiting", "we", "they", "our", "their", "not", "have"
    }
    ID_INDICATORS = {
        "yang", "dan", "di", "ke", "ini", "itu", "dengan", "untuk", "dari",
        "enak", "makan", "tempat", "sangat", "banget", "saya", "kami", "mereka",
        "tidak", "sudah", "belum", "bisa", "ada", "juga", "harga", "porsi"
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect the default natural language of the review (en, id, or jv)."""
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        if not tokens:
            return "id"
        en_matches = len(tokens & cls.EN_INDICATORS)
        id_matches = len(tokens & cls.ID_INDICATORS)

        if en_matches > id_matches:
            return "en"
        return "id"

    @classmethod
    def clean_and_normalize(cls, raw_text: str) -> Dict[str, Any]:
        """
        Anonymizes PII and extracts keyword indicators while strictly preserving
        the default original language of the customer review.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {
                "teks_asli": "",
                "bahasa": "id",
                "teks_anonim": "",
                "teks_bersih": "",
                "word_count": 0,
                "mention_qris": False,
                "mention_pos": False,
                "mention_edc": False,
                "mention_online": False,
            }

        # 1. Preserve original text & detect natural language
        raw_text_clean = raw_text.strip()
        lang = cls.detect_language(raw_text_clean)

        # 2. Anonymize PII (phone, email, URLs, mentions) while keeping language intact
        anonymized = TextAnonymizer.anonymize_text(raw_text_clean)

        # 3. Clean and tokenize
        lower_text = anonymized.lower()
        cleaned_symbols = re.sub(r"[^\w\s]", " ", lower_text)
        words = cleaned_symbols.split()

        # 4. Slang normalization ONLY for Indonesian/Javanese (Do NOT touch English reviews)
        if lang == "en":
            # Keep original English words without slang substitution
            normalized_words = words
            normalized_sentence = " ".join(words)
        else:
            normalized_words = [cls.SLANG_DICT.get(w, w) for w in words]
            normalized_sentence = " ".join(normalized_words)

        # 5. Multilingual Keyword indicator detection
        # Match against normalized lowercase text
        search_target = f" {normalized_sentence} "
        signals = {
            "mention_qris": any(f" {k} " in search_target or k in normalized_words for k in cls.DIGITAL_KEYWORDS["qris"]),
            "mention_pos": any(f" {k} " in search_target or k in normalized_words for k in cls.DIGITAL_KEYWORDS["pos"]),
            "mention_edc": any(f" {k} " in search_target or k in normalized_words for k in cls.DIGITAL_KEYWORDS["edc"]),
            "mention_online": any(f" {k} " in search_target or k in normalized_words for k in cls.DIGITAL_KEYWORDS["merchant_online"]),
        }

        return {
            "teks_asli": raw_text_clean,
            "bahasa": lang,
            "teks_anonim": anonymized,
            "teks_bersih": normalized_sentence,
            "word_count": len(words),
            **signals,
        }
