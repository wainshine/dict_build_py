# dict_build_py 修复任务

## 项目背景

**位置**: `/Users/wainshine/newAI/dict_build_py`

**用途**: 从原始中文文本中自动发现新词，构建领域/行业专属词典。输入一个语料库目录，输出 `词 词频 互信息 左右熵 位置成词概率 词性` 的词典表。

**来源**: 基于 Java 项目 [dict_build](https://github.com/sing1ee/dict_build) 用 Python 3.10+ 重构。算法思路来自 Matrix67 [博客](https://matrix67.com/blog/archives/5044)。本地 Java 原版代码在 `/Users/wainshine/newAI/dict_build`（如果 GitHub 访问不了可参考）。

**测试数据**:
- `/Users/wainshine/newAI/yidict/xiachufang/xiachufang.txt` — 1.5GB MySQL dump，大单文件
- `/Users/wainshine/newAI/yidict/gudaiwenxian/道藏` — 1721 个 .txt
- `/Users/wainshine/newAI/yidict/gudaiwenxian/诗藏` — 777 个 .txt
- `/Users/wainshine/newAI/yidict/gudaiwenxian/医藏` — 911 个 .txt

**运行方式**:
```bash
cd /Users/wainshine/newAI/dict_build_py
python3 -m dict_build <input_path> [--workers N] [--mem M]
```

**安装**: `pip install -e .`（已在当前环境安装）

**单元测试**: `python3 -m pytest tests/test_extract.py -v`（9 个，全部通过）

## 项目结构

```
dict_build_py/
├── pyproject.toml
├── .gitignore
├── README.md
├── dict_build/
│   ├── __init__.py
│   ├── __main__.py       # CLI (click)
│   ├── config.py         # 常量、阈值、停用词
│   ├── preprocess.py     # 文本清洗、中文分句 (regex)
│   ├── ngram.py          # N-gram 生成
│   ├── entropy.py        # 频率 + 左右熵 + 合并 (subprocess sort)
│   ├── pmi.py            # PMI + 过滤 (marisa-trie + mmap)
│   ├── pos_prob.py       # 位置成词概率加载
│   ├── pos_tag.py        # 词性标注 (jieba 词典查词)
│   ├── pipeline.py       # 流程编排 (multiprocessing)
│   └── data/pos_prop.txt # 位置成词概率 (~14K 行)
└── tests/
    └── test_extract.py   # 9 个 smoke test
```

**依赖**: marisa-trie, tqdm, click, jieba, regex（见 pyproject.toml）

## 关键架构决策

1. **外部排序**: 直接调用系统 `sort` 命令 + `LC_ALL=C`，不自己实现
2. **频率字典**: marisa-trie 构建后 save + mmap 加载，避免 Python 堆内存膨胀
3. **单行大文件**: 自动检测（>100MB 且 ≤10 行），走串行直写路径，跳过 multiprocessing 序列化开销
4. **N-gram 生成**: 正向（右熵）+ 反向（左熵），生成 `word\tnext_char` 格式
5. **合并**: 右熵文件天然有序，左熵文件反转后乱序需 `sort_file_inplace` 重排，然后归并 merge

## 需要修复的问题

### 问题 1 [CRITICAL] CLI 阈值参数无效

**根因**: Python 的 `from X import Y` 在 import 时绑定值，后续修改 `X.Y` 不影响已绑定的 `Y`。

**涉及文件**:
- `__main__.py:58-61` 修改 `config.PMI_THRESHOLD` 等
- `pipeline.py:22` `from .config import ENTROPY_THRESHOLD` — 永远是用默认值
- `pmi.py:88-91` `def extract_words(..., pmi_threshold=PMI_THRESHOLD, ...)` — 默认参数在函数定义时求值

**修复方案**: 让 `extract_words` 和 `merge_entropy_files_sorted` 不从模块级 import 阈值，改为在函数内部直接 `import dict_build.config as cfg` 读取 `cfg.PMI_THRESHOLD`（call-time 求值）。或者在 `run_pipeline` 中显式传参。

具体要改的地方:
1. `pmi.py` 的 `extract_words()` — 删除默认参数中的 `PMI_THRESHOLD`/`ENTROPY_THRESHOLD`/`POS_PROB_THRESHOLD`，改为函数内 `from .config import ...` 读取
2. `pipeline.py` 的 `run_pipeline()` — 调用 `merge_entropy_files_sorted` 时不依赖 import 的 `ENTROPY_THRESHOLD`，显式传入 `cfg.ENTROPY_THRESHOLD` 或函数内读取
3. `pipeline.py` 的 `run_pipeline()` — 调用 `extract_words` 时显式传入阈值

### 问题 2 [CRITICAL] Pool 多进程是串行的

**位置**: `pipeline.py:262-275` `_generate_ngrams_parallel()`

```python
for chunk in _read_chunks_by_lines(txt_file, CHUNK_LINES):
    fut = pool.apply_async(_process_chunk, (chunk, max_len))
    fw, bw = fut.get()   # ← 立即阻塞，下一个 chunk 提交前必须等这个完成
```

**修复方案**: 先收集所有 futures，再逐个 `get()`:
```python
futures = []
for chunk in _read_chunks_by_lines(txt_file, CHUNK_LINES):
    futures.append(pool.apply_async(_process_chunk, (chunk, max_len)))
for fut in futures:
    fw, bw = fut.get()
    fw_batch.extend(fw)
    bw_batch.extend(bw)
    ...
```

或者用 `pool.imap_unordered()` 更简洁。

### 问题 3 [MEDIUM] entropy.py 未处理 \r

**位置**: `entropy.py:32`

```python
suffix = line[tab_pos + 1:-1] if line.endswith("\n") else line[tab_pos + 1:]
```

`\r\n` 换行时 `line.endswith("\n")` 为 True，但 `line[:-1]` 只去掉 `\n`，`\r` 留在 suffix 里，污染熵计数。

**修复**: 改为 `line.rstrip("\n\r")` 或直接用 `line.strip()`。

### 问题 4 [MEDIUM] pos_prop.txt 未打包进 wheel

**位置**: `pyproject.toml:26-27`

只有 `packages = ["dict_build"]`，`dict_build/data/pos_prop.txt` 不会被打包。

**修复**: 添加:
```toml
[tool.setuptools.package-data]
"dict_build" = ["data/pos_prop.txt"]
```

### 问题 5 [MEDIUM] UTF-8 解码失败回退不完整

**位置**: `pipeline.py:339-345`

```python
for cut in range(-1, -5, -1):
    try:
        text = segment[:cut].decode("utf-8")
        carryover = segment[cut:]
        break
    except UnicodeDecodeError:
        continue
# 如果 4 次全失败，text/carryover 未绑定 → UnboundLocalError
```

**修复**: for 循环后加 `else: text, carryover = segment.decode("utf-8", errors="replace"), b""`

### 问题 6 [LOW] 死代码

| 位置 | 内容 |
|------|------|
| `preprocess.py:23-28` | `preprocess_file_lines()` 从未被调用 |
| `pmi.py:129-131` | `_tqdm()` 包装函数不必要 |
| `config.py:19` | `POS_PROP_PATH` 常量从未被读取 |

可以安全删除。

### 问题 7 [LOW] 其他小问题

- `pipeline.py:279-280` `pool.close()` / `pool.join()` 多余（`with Pool` 自动管理）
- `.gitignore` 缺少 `venv/`、`.coverage`、`*.log`
- `pos_tag.py:23` 直接读 jieba 内部属性 `pseg.dt.word_tag_tab`，升级可能炸
- 全项目无 logging 框架，全用 `print()`
- 测试只覆盖底层函数，pipeline/merge/POS tagging 无测试

## 验证方法

修复后运行以下命令验证:

```bash
# 1. 单元测试必须通过
python3 -m pytest tests/test_extract.py -v

# 2. 用 诗藏 验证 --min-freq 参数确实生效
python3 -m dict_build /Users/wainshine/newAI/yidict/gudaiwenxian/诗藏 --min-freq 5
# 预期: 结果数明显多于 --min-freq 10 的情况

# 3. 用 道藏 验证完整流程 + 输出行数正常
python3 -m dict_build /Users/wainshine/newAI/yidict/gudaiwenxian/道藏
# 预期: ~7.1 万词, 耗时正常

# 4. 验证输出格式
head -3 道藏/道藏_*_words_sort.data
# 预期: word\tfreq\tpmi\tentropy\tpos_prob\tpos 表头 + 数据

# 5. 检查 Pool 并行是否生效 (应快于旧版)
python3 -m dict_build /Users/wainshine/newAI/yidict/xiachufang/xiachufang.txt --workers 8
```

## 预期修复后的参考值

| 语料 | 文件数 | 大小 | 预期词数 |
|------|--------|------|---------|
| xiachufang.txt | 1 | 1.5GB | ~135K |
| 道藏 | 1721 | 132MB | ~71K |
| 诗藏 | 777 | 323MB | ~332K |
| 医藏 | 911 | 320MB | ~133K |

词数会根据 `--min-freq` 参数变化，以上为 `--min-freq 10` 的参考值。
