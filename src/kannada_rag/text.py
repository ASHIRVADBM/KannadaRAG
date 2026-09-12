"""Kannada-aware text normalisation and tokenisation.

Two distinct jobs live here and they must not be confused:

``normalise_for_index``
    Applied to corpus text before chunking and embedding, and to queries
    before encoding. Its purpose is to make semantically identical Kannada
    strings *identical byte sequences*, so that the embedding model does not
    see the same grapheme as two different token sequences depending on
    whether it came from a digital text layer or from OCR.

``tokenise_for_metrics``
    Applied only inside evaluation. Its purpose is to produce a token sequence
    on which n-gram overlap metrics behave sensibly for an agglutinative
    script. It is *not* used anywhere in the retrieval or generation path.

Keeping these separate matters: a normalisation that is appropriate for
scoring (e.g. stripping all non-Kannada characters) would silently destroy
dates, inscription numbers and Latin-script proper nouns if it were applied to
the corpus.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

__all__ = [
    "KANNADA_BLOCK",
    "normalise_for_index",
    "tokenise_for_metrics",
    "kannada_ratio",
    "is_predominantly_kannada",
    "character_ngrams",
]

# Kannada Unicode block.
KANNADA_BLOCK = (0x0C80, 0x0CFF)

# Zero-width joiner and non-joiner carry conjunct information in Kannada and
# must never be blanket-stripped from indexed text. Named here so they are
# visible in the source rather than being invisible literals.
ZWNJ = "\u200c"
ZWJ = "\u200d"

_KANNADA_RE = re.compile(r"[ಀ-೿]")
_KANNADA_RUN_RE = re.compile(r"[ಀ-೿\u200c\u200d]+")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RUN_RE = re.compile(r"[0-9೦-೯]+")

# Control and format characters that OCR engines sprinkle through Kannada
# output at syllable boundaries. ZWJ/ZWNJ are handled separately because they
# are meaningful inside conjunct clusters and must not be blanket-stripped
# from indexed text.
# Built from explicit code points rather than written as literal control
# characters, so the source file itself stays clean 7-bit-safe text.
_JUNK_CODEPOINTS = (
    list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20))
    + [0x7F, 0xFEFF, 0x00AD, 0x200B, 0x2060]
)
_JUNK_CONTROL_RE = re.compile(
    "[" + "".join(re.escape(chr(cp)) for cp in _JUNK_CODEPOINTS) + "]"
)
# Horizontal whitespace variants, again built from code points so that the
# source file contains no invisible literals.
_SPACE_CODEPOINTS = (
    [0x20, 0x09, 0x00A0, 0x202F, 0x205F, 0x3000] + list(range(0x2000, 0x200B))
)
_SOFT_SPACE_RE = re.compile(
    "[" + "".join(re.escape(chr(cp)) for cp in _SPACE_CODEPOINTS) + "]+"
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Danda and double danda are sentence punctuation; OCR frequently emits the
# Devanagari codepoints, a pipe, or a Latin full stop for the same mark.
_DANDA_VARIANTS = {
    "॥": "॥",  # double danda, canonical
    "।": "।",  # danda, canonical
    "||": "॥",
    "|": "।",
}


def normalise_for_index(text: str, form: str = "NFC") -> str:
    """Normalise corpus or query text prior to embedding.

    Steps, in order:

    1. Unicode canonical normalisation (NFC by default) so that a Kannada
       grapheme composed of a base consonant plus a combining vowel sign is
       encoded identically regardless of the producing software.
    2. Removal of control and zero-width-space noise characters, while
       preserving ZWJ (U+200D) and ZWNJ (U+200C), which carry conjunct
       information in Kannada.
    3. Canonicalisation of danda punctuation variants.
    4. Whitespace collapsing that preserves paragraph structure, because
       paragraph boundaries are the highest-priority chunk separator.
    """
    if not text:
        return ""

    text = unicodedata.normalize(form, text)
    text = _JUNK_CONTROL_RE.sub("", text)

    for variant, canonical in _DANDA_VARIANTS.items():
        if variant in ("|", "||"):
            text = text.replace(variant, canonical)

    text = _SOFT_SPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def kannada_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that lie in the Kannada block.

    Whitespace is excluded from the denominator so that a correctly written
    Kannada sentence is not penalised for having spaces between words.
    """
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    return len(_KANNADA_RE.findall(stripped)) / len(stripped)


def is_predominantly_kannada(text: str, threshold: float = 0.5) -> bool:
    """Language guard used on incoming queries.

    Note this is a *ratio* test, not a presence test. A query containing a
    single Kannada character inside an otherwise English sentence must be
    rejected, because the downstream instruction-tuned models will otherwise
    answer in English and defeat the monolingual design of the system.
    """
    return kannada_ratio(text) >= threshold


def tokenise_for_metrics(text: str, keep_numerals: bool = True) -> list[str]:
    """Tokenise for ROUGE / METEOR scoring.

    Kannada is agglutinative and written without inter-morpheme boundaries, so
    whitespace tokens are long and surface-level n-gram overlap is
    systematically pessimistic. This tokeniser:

    * normalises to NFC and casefolds Latin runs;
    * emits Kannada runs, Latin runs and numeral runs as separate tokens,
      rather than deleting non-Kannada characters outright -- dates, regnal
      years and inscription identifiers are load-bearing content in heritage
      answers and deleting them inflates scores;
    * discards punctuation.

    ``keep_numerals=False`` reproduces a Kannada-only variant for ablation.
    """
    if not text:
        return []

    text = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    for match in re.finditer(
        r"[ಀ-೿\u200c\u200d]+|[A-Za-z]+|[0-9೦-೯]+", text
    ):
        token = match.group(0)
        if _LATIN_RUN_RE.fullmatch(token):
            tokens.append(token.casefold())
        elif _DIGIT_RUN_RE.fullmatch(token):
            if keep_numerals:
                tokens.append(token)
        else:
            # Strip ZWJ/ZWNJ for scoring only; they do not change the
            # pronounced form and their presence is OCR-dependent.
            tokens.append(token.replace(ZWNJ, "").replace(ZWJ, ""))
    return [t for t in tokens if t]


def character_ngrams(text: str, n: int = 4) -> list[str]:
    """Character n-grams over normalised text.

    Provided as a morphology-robust complement to word-level overlap: two
    Kannada forms of the same lemma that differ only in a case suffix share
    most of their character n-grams but zero word unigrams.
    """
    cleaned = "".join(tokenise_for_metrics(text))
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(tokens)
