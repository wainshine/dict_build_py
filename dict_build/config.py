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
CHUNKS_PER_BATCH = 20
SENTINEL = "$"

# Bucket-sort thresholds: n-gram files larger than this get partitioned
BUCKET_SORT_MIN_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
BUCKET_TARGET_BYTES = 4 * 1024 * 1024 * 1024     # 4 GB per bucket
MIN_SORT_MEM_MB = 512                              # minimum MB per sort process

STOPWORDS = set("的很了么呢是嘛个都也比还这于不与才上用就好在和对挺去后没说")

# Known encoding-artifact word fragments (e.g. GBK↔UTF-8 double-miscoding)
_BLOCKLIST_PIECES = ["锟斤拷", "拷斤拷", "烫烫烫", "屯屯屯", "铪铪", "锟斤", "斤拷"]
WORD_BLOCKLIST_SUBSTRINGS = tuple(_BLOCKLIST_PIECES)
