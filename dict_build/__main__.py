"""CLI entry point for dict_build.

Usage:
    python -m dict_build <input>            # auto output path
    python -m dict_build <input> -o out.txt # custom output
    python -m dict_build /path/to/corpus.txt
    python -m dict_build /path/to/dir/
"""

import click

from .config import (
    DEFAULT_MAX_LEN, DEFAULT_MEM_MB, WORKERS, MIN_FREQ,
    PMI_THRESHOLD, ENTROPY_THRESHOLD, POS_PROB_THRESHOLD,
)
from .pipeline import run_pipeline


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True, readable=True))
@click.option("--output", "-o", default=None,
              help="Output file path (auto-generated if omitted).")
@click.option("--max-len", "-l", default=DEFAULT_MAX_LEN,
              help="Maximum word length for n-gram generation.")
@click.option("--mem", "-m", default=DEFAULT_MEM_MB,
              help="Memory budget in MB for external sort.")
@click.option("--workers", "-w", default=WORKERS,
              help="Number of worker processes.")
@click.option("--min-freq", default=MIN_FREQ,
              help="Minimum frequency for trie inclusion.")
@click.option("--pos-prop", default=None,
              help="Path to pos_prop.txt.")
@click.option("--pmi-threshold", default=PMI_THRESHOLD,
              help="Minimum PMI value.")
@click.option("--entropy-threshold", default=ENTROPY_THRESHOLD,
              help="Minimum entropy value.")
@click.option("--pos-threshold", default=POS_PROB_THRESHOLD,
              help="Minimum position probability.")
def main(
    input_path: str,
    output: str | None,
    max_len: int,
    mem: int,
    workers: int,
    min_freq: int,
    pos_prop: str | None,
    pmi_threshold: float,
    entropy_threshold: float,
    pos_threshold: float,
) -> None:
    """Extract Chinese words/phrases from raw text using statistical NLP.

    INPUT_PATH can be a .txt file or a directory containing .txt files.

    Output format (tab-separated, sorted by frequency desc):
        word    frequency    pmi    entropy    position_probability
    """
    import dict_build.config as cfg
    cfg.PMI_THRESHOLD = pmi_threshold
    cfg.ENTROPY_THRESHOLD = entropy_threshold
    cfg.POS_PROB_THRESHOLD = pos_threshold

    run_pipeline(
        input_path=input_path,
        output_path=output,
        max_len=max_len,
        mem_mb=mem,
        workers=workers,
        min_freq=min_freq,
        pos_prob_path=pos_prop,
    )


if __name__ == "__main__":
    main()
