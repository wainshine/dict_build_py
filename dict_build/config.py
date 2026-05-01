"""Configuration constants for dict_build."""

import regex as re
from multiprocessing import cpu_count

OUTPUT_FILE_SUFFIX = "words_sort.data"
DEFAULT_MAX_LEN = 6
DEFAULT_MEM_MB = 4096
MIN_FREQ = 10
PMI_THRESHOLD = 1.0
ENTROPY_THRESHOLD = 2.0
POS_PROB_THRESHOLD = 0.1
WORKERS = max(1, cpu_count() // 2)
CHUNK_LINES = 20_000
WRITE_BATCH = 50_000
SENTINEL = "$"
CHINESE_MIN = 0x4E00
CHINESE_MAX = 0x9FA5

STOPWORDS = set("的很了么呢是嘛个都也比还这于不与才上用就好在和对挺去后没说")

PUNCT_RE = re.compile(r"[\p{P}\p{S}\p{Z}\p{C}\p{M}]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]+")
NON_CHINESE_RE = re.compile(r"[^\u4e00-\u9fa5]")

POS_PROP_PATH = "data/pos_prop.txt"
