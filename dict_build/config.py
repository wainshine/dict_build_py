"""Configuration constants for dict_build."""

from multiprocessing import cpu_count

OUTPUT_FILE_SUFFIX = "words_sort.data"
DEFAULT_MAX_LEN = 6
DEFAULT_MEM_MB = 4096
MIN_FREQ = 10
PMI_THRESHOLD = 1.0
ENTROPY_THRESHOLD = 2.0
POS_PROB_THRESHOLD = 0.1
WORKERS = max(1, cpu_count() // 2)
CHUNK_LINES = 5_000
SENTINEL = "$"

STOPWORDS = set("的很了么呢是嘛个都也比还这于不与才上用就好在和对挺去后没说")
