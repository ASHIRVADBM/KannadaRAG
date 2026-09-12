"""Prompt templates, held constant across every experimental condition.

The grounded and ungrounded prompts are deliberately as close to identical as
the conditions permit: same language, same length instruction, same register.
Only the context block and the sentence licensing its use differ. If the two
prompts differed in tone or verbosity, a measured difference between the RAG
and no-RAG conditions could be attributed to prompt wording rather than to
retrieval, and the controlled comparison would not be controlled.
"""

from __future__ import annotations

__all__ = [
    "GROUNDED_PROMPT",
    "UNGROUNDED_PROMPT",
    "ABSTENTION_RESPONSE",
    "NON_KANNADA_RESPONSE",
    "build_prompt",
]


GROUNDED_PROMPT = """ನೀವು ಕರ್ನಾಟಕದ ಪಾರಂಪರಿಕ ತಾಣಗಳ ಕುರಿತ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವ ಸಹಾಯಕ.

ಕೆಳಗೆ ನೀಡಿರುವ ಆಧಾರ ಪಠ್ಯದಲ್ಲಿ ಇರುವ ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ ಉತ್ತರಿಸಿ.
ಆಧಾರ ಪಠ್ಯದಲ್ಲಿ ಇಲ್ಲದ ಯಾವುದೇ ಮಾಹಿತಿಯನ್ನು ಸೇರಿಸಬೇಡಿ.
ಆಧಾರ ಪಠ್ಯದಲ್ಲಿ ಉತ್ತರ ಸಿಗದಿದ್ದರೆ "ಈ ಮಾಹಿತಿ ದಾಖಲೆಯಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ" ಎಂದು ಮಾತ್ರ ಹೇಳಿ.
ಸಂಪೂರ್ಣ ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿಯೇ ಬರೆಯಿರಿ.
ಉತ್ತರವು {n_sentences} ವಾಕ್ಯಗಳಲ್ಲಿ ಇರಲಿ.

ಆಧಾರ ಪಠ್ಯ:
{context}

ಪ್ರಶ್ನೆ: {question}

ಉತ್ತರ:"""


UNGROUNDED_PROMPT = """ನೀವು ಕರ್ನಾಟಕದ ಪಾರಂಪರಿಕ ತಾಣಗಳ ಕುರಿತ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವ ಸಹಾಯಕ.

ನಿಮಗೆ ತಿಳಿದಿರುವ ಮಾಹಿತಿಯನ್ನು ಬಳಸಿ ಉತ್ತರಿಸಿ.
ಉತ್ತರ ತಿಳಿಯದಿದ್ದರೆ "ಈ ಮಾಹಿತಿ ನನಗೆ ತಿಳಿದಿಲ್ಲ" ಎಂದು ಮಾತ್ರ ಹೇಳಿ.
ಸಂಪೂರ್ಣ ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿಯೇ ಬರೆಯಿರಿ.
ಉತ್ತರವು {n_sentences} ವಾಕ್ಯಗಳಲ್ಲಿ ಇರಲಿ.

ಪ್ರಶ್ನೆ: {question}

ಉತ್ತರ:"""


ABSTENTION_RESPONSE = "ಈ ಮಾಹಿತಿ ದಾಖಲೆಯಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ."

NON_KANNADA_RESPONSE = (
    "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ಕೇಳಿ. "
    "ಈ ವ್ಯವಸ್ಥೆ ಕನ್ನಡ ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಉತ್ತರಿಸುತ್ತದೆ."
)


def build_prompt(
    question: str,
    context: str | None,
    n_sentences: str = "2-3",
) -> str:
    """Construct the prompt for either condition.

    ``context is None`` selects the ungrounded template. The caller, not this
    function, decides which condition is being run, so that the two paths stay
    symmetric and neither can accidentally acquire an extra instruction.
    """
    if context is None:
        return UNGROUNDED_PROMPT.format(question=question, n_sentences=n_sentences)
    return GROUNDED_PROMPT.format(
        question=question, context=context, n_sentences=n_sentences
    )
