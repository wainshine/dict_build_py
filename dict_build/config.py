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

# Backpressure: cap in-flight pool tasks at workers * PENDING_TASK_FACTOR
PENDING_TASK_FACTOR = 4

# Batched n-gram write buffer (entries, not bytes)
NGRAM_WRITE_BATCH = 65536

# Rough line count estimate: file_size // AVG_LINE_BYTES
AVG_LINE_BYTES = 100

# Bucket-sort thresholds: n-gram files larger than this get partitioned
BUCKET_SORT_MIN_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
BUCKET_TARGET_BYTES = 4 * 1024 * 1024 * 1024     # 4 GB per bucket
MIN_SORT_MEM_MB = 512                              # minimum MB per sort process
MAX_BUCKETS = 64        # cap: distribution opens 2 x num_buckets handles per batch
BUCKET_DISTRIBUTE_BATCH = 15   # temp files per distribution batch
BUCKET_BUF_MIN_BYTES = 128 * 1024 * 1024   # per-bucket buffer floor
BUCKET_BUF_MAX_BYTES = 256 * 1024 * 1024   # per-bucket buffer cap

# Sorted n-gram files larger than this get split for parallel entropy
PARALLEL_ENTROPY_MIN_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
ENTROPY_SPLIT_CHARS_PER_FILE = 10

# Estimated n-gram intermediate size as a multiple of input size
# (forward + backward n-grams, sorted copies, entropy files)
INTERMEDIATE_SIZE_FACTOR = 8

# Huge single-line file detection: files larger than this with very few
# newlines in head/mid/tail samples are processed in streaming mode
SINGLE_LINE_MIN_BYTES = 100 * 1024 * 1024
SINGLE_LINE_MAX_NEWLINES = 10
SINGLE_LINE_SAMPLE_BYTES = 4 * 1024 * 1024

# Encoding detection: per-sample size and CJK ratio heuristics
ENCODING_SAMPLE_BYTES = 64 * 1024
CJK_RATIO_UTF8_MIN = 0.005   # recovered-CJK threshold to keep UTF-8
CJK_RATIO_WINNER_MIN = 0.01  # clear-winner threshold for GBK-family

# In-memory sort fallback limit when no system sort exists
FALLBACK_SORT_MAX_BYTES = 256 * 1024 * 1024

STOPWORDS = set("的很了么呢是嘛个都也比还这于不与才上用就好在和对挺去后没说")

# Known encoding-artifact word fragments (e.g. GBK↔UTF-8 double-miscoding)
_BLOCKLIST_PIECES = ["锟斤拷", "拷斤拷", "烫烫烫", "屯屯屯", "铪铪", "锟斤", "斤拷"]
WORD_BLOCKLIST_SUBSTRINGS = tuple(_BLOCKLIST_PIECES)
