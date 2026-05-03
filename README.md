# dict_build_py

版本 1.1.0 | 作者 wainshine | 协议 Apache-2.0

从原始文本中自动发现中文新词，构建领域/行业专属词典。

基于 [dict_build](https://github.com/sing1ee/dict_build)（Java 版）用 Python 3.10+ 完整重构，算法核心和位置成词概率文件（`pos_prop.txt`）均源于此；成词判定的方法论思路来自 Matrix67 的博客文章[《互联网时代的社会语言学：基于SNS的文本数据挖掘》](https://matrix67.com/blog/archives/5044)。

重构后增加了多进程并行、大单行文件处理、词性标注、编码自动检测、熵计算并行化等能力。

## 安装

```bash
cd dict_build_py
pip install -e .
```

依赖：Python 3.10+、`marisa-trie`、`tqdm`、`click`、`jieba`、`regex`、`charset-normalizer`。

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
| `input_path` | 必填 | .txt 文件或目录 |
| `--output, -o` | 自动 | 输出路径（默认：`{原名}_{日期}_words_sort.data`） |
| `--max-len, -l` | 6 | N-gram 最大词长 |
| `--mem, -m` | 4096 | 排序内存，MB |
| `--workers, -w` | CPU核数/2 | 工作进程数 |
| `--min-freq` | 10 | 频率字典最低词频 |
| `--pos-prop` | 自动 | 位置成词概率文件路径 |
| `--pmi-threshold` | 1.0 | 互信息阈值 |
| `--entropy-threshold` | 2.0 | 熵阈值 |
| `--pos-threshold` | 0.1 | 位置成词概率阈值 |

### 支持的文件格式与编码

格式：`.txt` `.csv` `.json` `.sql` `.md` `.html` `.htm`

编码：UTF-8 / GBK / GB18030 自动检测，无需手动转码。

### 自动适配三种场景

| 场景 | 处理方式 |
|------|---------|
| 大单文件（多行） | 多进程分块，worker 直写临时文件（避 pickle 序列化），主进程拼接 |
| 目录（多个文件） | 递归扫描，同上 |
| 单行超大文件 | 自动检测（>100MB 且 ≤10 行），串行分块直写，UTF-8 边界回退容错 |

> N-gram 阶段采用背压限流（在途任务 ≤ `workers × 4`），防止 Pool 队列积压；熵计算在 n-gram 文件 >1GB 时自动走并行拆分路径；CLI 阈值参数在函数调用时从 config 读取，传参必定生效。

## 成词判定

满足全部四个条件，视为一个新词：

| 指标 | 说明 | 默认阈值 |
|------|------|---------|
| 互信息 (PMI) | 字与字之间的粘合度。越高越像一个固定的词 | >= 1.0 |
| 左右熵 | 词的左右邻字多样性。熵越高，词的边界越清晰 | >= 2.0 |
| 位置成词概率 | 首字出现在词首的概率 × 尾字出现在词尾的概率 | >= 0.1 |
| N-gram 频率 | 词在语料中的总出现次数 | >= 10 |

## 算法流程

```
输入文件
  │
  ├─ 预处理: 正则清洗 + 停用词过滤 + 中文分句 + 哨兵填充
  │
  ├─ N-gram 生成: 正向(右熵) + 反向(左熵), 1~max_len 字滑动窗口
  │    └─ 大语料自动并行：按首字拆分 → Pool 分布式计算 → 拼接
  │
  ├─ 外部排序: 系统 sort 命令（LC_ALL=C 字节序，并行，带进度）
  │
  ├─ 频率 + 左右熵: 按首字分组流式计算 Shannon 熵
  │    └─ n-gram 文件 >1GB 自动拆分并行
  │
  ├─ 熵合并: 取 min(左熵, 右熵)
  │
  ├─ PMI + 位置概率: marisa-trie (mmap) 前缀查询 + pos_prop.txt 查表
  │
  └─ 词性标注: jieba 词典直接匹配, O(1) 查词
```

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

## 使用场景

输入一个领域的语料库，输出该领域的专属词典。不同语料产出的高频词差异鲜明：

| 语料 | 规模 | 编码 | 产词 | 典型高频词 |
|------|------|------|------|-----------|
| 道藏 | 132MB / 1721 文件 | UTF-8 | 71,190 | 天下、天地、真人、阴阳 |
| 诗藏 | 336MB / 776 文件 | UTF-8 | 331,696 | 一作、万里、先生、今日 |
| 医藏 | 320MB / 789 文件 | UTF-8 | 133,156 | 甘草、人参、一味、小便 |
| 易藏 | 100MB / 343 文件 | UTF-8 | 102,685 | 君子、天下、故曰、象曰 |
| 下厨房菜谱 | 1.5GB / 1 文件 | UTF-8 | 135,087 | 鸡蛋、黄油、烤箱、翻炒 |
| 地名 | 1.7GB / 2 文件 | GB18030 | 112,479 | 东路、大街、胡同、路口 |
| 兴趣点 | 2.0GB / 2 文件 | GBK | 309,448 | 酒店、大厦、广场、中心 |

人名本身就是一个词——哪个领域的词典收录了它，这个人名就带上了该领域的风格标签。例如"甘草"出现在医藏词典中，有中医/药草风格；"真人"出现在道藏词典中，有道家/修仙风格。这在取名、风格分析、标签推荐等场景中十分有用。

## 示例

**道藏** — 道家/修仙风格词典：

```
word	freq	pmi	entropy	pos_prob	pos
天下	39656	4.631669	5.688685	0.222117	s
天地	38915	5.140978	5.610884	0.281994	n
自然	29265	6.098053	5.595066	0.263994	d
真人	28890	3.789042	5.809989	0.358093	n
人之	28179	1.388978	6.200332	0.135896	x
```

**下厨房菜谱** — 烹饪/美食风格词典：

```
word	freq	pmi	entropy	pos_prob	pos
可以	1016014	6.271348	3.256264	0.136838	c
加入	913942	4.890289	3.316592	0.316673	v
放入	873827	4.794842	4.148527	0.307969	v
倒入	517635	5.540863	4.199828	0.316673	v
鸡蛋	498495	5.780279	2.714642	0.241632	n
```

**医藏** — 中医/药草风格词典：

```
word	freq	pmi	entropy	pos_prob	pos
甘草	137005	7.900791	2.574762	0.349703	n
一钱	126202	4.410848	2.283377	0.165264	x
人参	91426	6.763619	2.549888	0.135896	n
小便	81848	6.815504	4.852599	0.181202	nr
```

## 项目结构

```
dict_build_py/
├── pyproject.toml
├── LICENSE
├── dict_build/
│   ├── __init__.py
│   ├── __main__.py       # CLI 入口
│   ├── config.py         # 常量、阈值、停用词
│   ├── preprocess.py     # 文本清洗、中文分句
│   ├── ngram.py          # N-gram 生成
│   ├── entropy.py        # 频率 + 左右熵 + 合并
│   ├── pmi.py            # PMI + 过滤 (marisa-trie)
│   ├── pos_prob.py       # 位置成词概率加载
│   ├── pos_tag.py        # 词性标注 (jieba)
│   ├── pipeline.py       # 流程编排
│   └── data/pos_prop.txt # 位置成词概率
└── tests/
    └── test_extract.py   # 38 个单元测试
```

## 与 dict_build (Java) 版本的差异

| 方面 | dict_build (Java) | dict_build_py (Python) |
|------|------------------|----------------------|
| 并行处理 | 单线程 | 多进程 N-gram + sort 并行 + 熵并行 |
| 大单行文件 | 不支持 | 自动检测 + 串行直写 |
| 编码检测 | 仅 UTF-8 | UTF-8 / GBK / GB18030 自动嗅探 |
| 多格式输入 | 仅 .txt | .txt .csv .json .sql .md .html .htm |
| 词性标注 | 无 | jieba 词典 O(1) 查词 |
| 进度提示 | LOG 信息 | tqdm 全阶段进度条 |
| 内存优化 | ConcurrentRadixTree | marisa-trie mmap + worker 直写 |
| 输出位置 | 原文件同目录 | 同目录 + 日期后缀 |
| 算法核心 | N-gram + PMI + 左右熵 + 位置概率 | 完全一致 |

## 致谢

- [dict_build](https://github.com/sing1ee/dict_build) — Java 原版项目，算法和 pos_prop.txt 的来源
- [Matrix67](https://matrix67.com/blog/archives/5044) — 成词判定方法论的思路源头
