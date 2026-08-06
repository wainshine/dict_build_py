"""CLI entry point for dict_build.

Usage:
    python -m dict_build <input>            # auto output path
    python -m dict_build <input> -o out.txt # custom output
    python -m dict_build /path/to/corpus.txt
    python -m dict_build /path/to/dir/
"""

import logging

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
@click.option("--temp-dir", default=None,
              type=click.Path(exists=True, file_okay=False, writable=True),
              help="Parent directory for temporary intermediate files "
                   "(default: system temp, override with TMPDIR env var).")
@click.option("--work-dir", default=None,
              type=click.Path(file_okay=False),
              help="Persistent working directory: enables checkpointed "
                   "resume after interruption.")
@click.option("--force", is_flag=True,
              help="Ignore existing checkpoints in --work-dir and start over.")
@click.option("--verbose", "-v", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", "-q", is_flag=True, help="Only errors and the "
              "progress bars.")
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
    temp_dir: str | None,
    work_dir: str | None,
    force: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Extract Chinese words/phrases from raw text using statistical NLP.

    INPUT_PATH can be a .txt file or a directory containing text files.

    Output format (tab-separated, sorted by frequency desc):
        word    freq    pmi    entropy    pos_prob    pos
    """
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    logging.basicConfig(format="%(message)s", level=level)

    run_pipeline(
        input_path=input_path,
        output_path=output,
        max_len=max_len,
        mem_mb=mem,
        workers=workers,
        min_freq=min_freq,
        pos_prob_path=pos_prop,
        temp_dir=temp_dir,
        work_dir=work_dir,
        force=force,
        pmi_threshold=pmi_threshold,
        entropy_threshold=entropy_threshold,
        pos_threshold=pos_threshold,
    )


if __name__ == "__main__":
    main()
