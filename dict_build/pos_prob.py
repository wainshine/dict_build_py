"""Load position-based word formation probability data.

pos_prop.txt format: char<TAB>P(S)<TAB>P(M)<TAB>P(E)
S = probability char appears at Start of word
M = probability char appears in Middle of word
E = probability char appears at End of word
"""

import os


def load_pos_prob(filepath: str | None = None) -> dict[str, tuple[float, float, float]]:
    """Load pos_prop.txt and return dict mapping char to (P_S, P_M, P_E)."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "data", "pos_prop.txt")

    result: dict[str, tuple[float, float, float]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            char = parts[0]
            p_s = float(parts[1])
            p_m = float(parts[2])
            p_e = float(parts[3])
            result[char] = (p_s, p_m, p_e)
    return result
