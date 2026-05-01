"""POS tagging for extracted Chinese words using jieba's dictionary.

For single words out of context, uses dictionary lookup.
Unknown words are marked as 'x'.
"""

from __future__ import annotations

import jieba.posseg as pseg


# Cache the POS dictionary (lazy loaded)
_pos_dict: dict[str, str] | None = None


def _get_pos_dict() -> dict[str, str]:
    """Get jieba's POS dictionary as {word: tag}."""
    global _pos_dict
    if _pos_dict is not None:
        return _pos_dict
    _pos_dict = {}
    # jieba.posseg.dt is a Trie that maps word -> (freq, tag)
    for word, tag in pseg.dt.word_tag_tab.items():
        _pos_dict[word] = tag
    return _pos_dict


def tag_word(word: str) -> str:
    """Return POS tag for a single word.

    Tags follow ICTCLAS standard:
        n=noun, v=verb, a=adjective, d=adverb, p=preposition,
        c=conjunction, u=auxiliary, m=numeral, q=quantifier,
        r=pronoun, x=unknown, etc.
    """
    return _get_pos_dict().get(word, "x")


def tag_words(words: list[str]) -> list[str]:
    """Tag a list of words, returning list of POS tags."""
    d = _get_pos_dict()
    return [d.get(w, "x") for w in words]
