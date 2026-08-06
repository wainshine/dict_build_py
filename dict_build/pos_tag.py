"""POS tagging for extracted Chinese words using jieba's dictionary.

For single words out of context, uses dictionary lookup.
Unknown words are marked as 'x'.

jieba is imported lazily on first use: spawned multiprocessing workers
never pay the import cost, and jieba's "Building prefix dict" log noise
is silenced.
"""

from __future__ import annotations

import logging


# Cache the POS dictionary (lazy loaded)
_pos_dict: dict[str, str] | None = None


def _get_pos_dict() -> dict[str, str]:
    """Get jieba's POS dictionary as {word: tag}."""
    global _pos_dict
    if _pos_dict is not None:
        return _pos_dict
    import jieba
    jieba.setLogLevel(logging.WARNING)
    import jieba.posseg as pseg
    # jieba.posseg.dt is a Trie that maps word -> (freq, tag)
    _pos_dict = dict(pseg.dt.word_tag_tab)
    return _pos_dict


def tag_word(word: str) -> str:
    """Return POS tag for a single word.

    Tags follow ICTCLAS standard:
        n=noun, v=verb, a=adjective, d=adverb, p=preposition,
        c=conjunction, u=auxiliary, m=numeral, q=quantifier,
        r=pronoun, x=unknown, etc.
    """
    return _get_pos_dict().get(word, "x")
