# dict_build_py

从原始文本中自动发现中文新词，构建领域/行业专属词典。

基于 [dict_build](https://github.com/sing1ee/dict_build)（Java 版）用 Python 3.10+ 完整重构，算法核心和位置成词概率文件 (`pos_prop.txt`) 均源于此；成词判定的方法论思路则来自 Matrix67 的博客文章[《互联网时代的社会语言学：基于SNS的文本数据挖掘》](https://matrix67.com/blog/archives/5044)。

重构后增加了多进程并行、大单行文件处理、词性标注、多格式输入等能力。

## 成词判定

满足全部四个条件，视为一个新词：

| 指标 | 说明 | 默认阈值 |
|------|------|---------|
| **互信息 (PMI)** | 字与字之间的粘合度。越高越像一个固定的词 | >= 1.0 |
| **左右熵** | 词的左右邻字多样性。熵越高，词的边界越清晰 | >= 2.0 |
| **位置成词概率** | 首字出现在词首的概率 × 尾字出现在词尾的概率 | >= 0.1 |
| **N-gram 频率** | 词在语料中的总出现次数 | >= 10 |

## 使用场景

以语料库为输入，构建该语料所属领域/行业的专有词典。

例如：

```bash
python -m dict_build /path/to/道藏/
```

输出 `道藏_20260501_words_sort.data`，就是一个**道家词典**。

如果一个人名匹配该词典中的词汇，则可判定该人名具有道家/修仙风格。这在取名、标签推荐、内容风格分析等场景中有用。

## 安装

```bash
cd dict_build_py
pip install -e .
```

依赖：Python 3.10+，`marisa-trie`、`tqdm`、`click`、`jieba`、`regex`。

## 用法

```bash
# 单文件
python -m dict_build /path/to/corpus.txt

# 目录（递归扫描子目录）
python -m dict_build /path/to/corpus_dir/

# 完整参数
python -m dict_build input.txt \
    -o output.txt \
    --max-len 6 \
    --mem 4096 \
    --workers 8 \
    --min-freq 10 \
    --pmi-threshold 1.0 \
    --entropy-threshold 2.0 \
    --pos-threshold 0.1
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_path` | 必填 | .txt 文件或包含文本文件的目录 |
| `--output, -o` | 自动 | 输出路径。默认格式：`{源文件名}_{日期}_words_sort.data` |
| `--max-len, -l` | 6 | N-gram 最大词长 |
| `--mem, -m` | 4096 | 排序可用内存，单位 MB |
| `--workers, -w` | CPU核数/2 | 并行工作进程数 |
| `--min-freq` | 10 | 最低词频，低于此值的词不进入频率字典 |
| `--pos-prop` | 自动 | 位置成词概率文件路径（pos_prop.txt） |
| `--pmi-threshold` | 1.0 | PMI 最低阈值 |
| `--entropy-threshold` | 2.0 | 熵最低阈值 |
| `--pos-threshold` | 0.1 | 位置成词概率最低阈值 |

### 支持的文件格式

`.txt` `.csv` `.json` `.sql` `.md` `.html` `.htm`

### 自动适配三种场景

| 场景 | 处理方式 |
|------|---------|
| **大单文件** | 多进程按行分块，并行预处理+N-gram |
| **目录（多个小文件）** | 递归扫描，同上处理 |
| **单行超大文件** | 自动检测（>100MB 且 ≤10 行），串行内存直写，避免序列化开销 |

## 输出格式

```
word	freq	pmi	entropy	pos_prob	pos
```

| 列 | 含义 |
|----|------|
| `word` | 词 |
| `freq` | 词频 |
| `pmi` | 互信息 |
| `entropy` | min(左熵, 右熵) |
| `pos_prob` | 位置成词概率 |
| `pos` | 词性（ICTCLAS 标准：n=名词, v=动词, a=形容词, d=副词, c=连词, x=未知…） |

按词频降序排列，第一行为表头。

### 示例

下厨房菜谱语料：

```
word	freq	pmi	entropy	pos_prob	pos
可以	1016014	6.271348	3.256264	0.136838	c
加入	913942	4.890289	3.316592	0.316673	v
放入	873827	4.794842	4.148527	0.307969	v
倒入	517635	5.540863	4.199828	0.316673	v
鸡蛋	498495	5.780279	2.714642	0.241632	n
```

道藏典籍语料：

```
word	freq	pmi	entropy	pos_prob	pos
天下	39656	4.631669	5.688685	0.222117	s
天地	38915	5.140978	5.610884	0.281994	n
自然	29265	6.098053	5.595066	0.263994	d
真人	28890	3.789042	5.809989	0.358093	n
人之	28179	1.388978	6.200332	0.135896	x
```

## 算法流程

```
输入文件
  │
  ├─ 预处理: 正则清洗 + 停用词过滤 + 中文分句 + 哨兵填充
  │
  ├─ N-gram 生成: 正向(右熵) + 反向(左熵), 1~max_len 字滑动窗口
  │
  ├─ 外部排序: 系统 sort 命令并行处理
  │
  ├─ 频率 + 左右熵: 按首字分组流式计算 Shannon 熵
  │
  ├─ 熵合并: 取 min(左熵, 右熵)
  │
  ├─ PMI + 位置概率: marisa-trie (mmap) 前缀查询 + pos_prop.txt 查表
  │
  └─ 词性标注: jieba 词典直接匹配, O(1) 查词
```

## 项目结构

```
dict_build_py/
├── pyproject.toml
├── dict_build/
│   ├── __init__.py
│   ├── __main__.py       # CLI 入口
│   ├── config.py         # 常量、阈值、停用词
│   ├── preprocess.py     # 文本清洗、中文分句
│   ├── ngram.py          # N-gram 生成
│   ├── entropy.py        # 频率 + 左右熵计算 + 合并
│   ├── pmi.py            # PMI 计算 + 过滤 (marisa-trie)
│   ├── pos_prob.py       # 位置成词概率加载
│   ├── pos_tag.py        # 词性标注 (jieba)
│   ├── pipeline.py       # 流程编排
│   └── data/pos_prop.txt # 位置成词概率数据
└── tests/
    └── test_extract.py   # 9 个单元测试
```

## 与 dict_build (Java) 版本的差异

| 方面 | dict_build (Java) | dict_build_py (Python) |
|------|------------------|----------------------|
| 并行处理 | 单线程 | 多进程 n-gram 生成, 并行 sort |
| 大单行文件 | 不支持 | 自动检测 + 串行直写 |
| 多格式输入 | 仅 .txt | .txt .csv .json .sql .md .html .htm |
| 词性标注 | 无 | jieba 词典 O(1) 查词 |
| 进度提示 | LOG 信息 | tqdm 多阶段进度条 |
| 内存优化 | ConcurrentRadixTree | marisa-trie + mmap 零拷贝加载 |
| 输出位置 | 原文件同目录 | 同目录 + 日期后缀 |
| 算法 | N-gram + PMI + 左右熵 + 位置概率 | 完全一致 |
