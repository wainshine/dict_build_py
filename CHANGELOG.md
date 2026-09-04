# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 简式约定。

> 发布新版本时，版本号需同步更新三处：`pyproject.toml`、`dict_build/__init__.py`（fallback）、`README.md` 头部，并在本文件追加条目。

## [1.4.2]

- 修复 BIG5 编码检测误判（P1）：BIG5 字节按 UTF-8+replace 解码会漏出约 2% 偶然 CJK，超过旧的 0.5% 阈值导致误判为 UTF-8；现增加 U+FFFD 比率约束（受损 UTF-8 的 FFFD 率 <1%，BIG5 伪装约 45%）
- 顺带修复：严格 UTF-8 解码成功但文件已含 U+FFFD 时，不再落入 GBK 候选比较（避免被误判为 gb18030）
- 新增 4 个测试：BIG5 检测、受损 UTF-8 检测、并行熵路径（`_write_entropy_from_ngram_parallel`）、extract_words 输出降序断言
- 测试 81→85
- 文档体系：版本历史迁至 CHANGELOG.md；新增 HANDOVER.md（会话交接）；README 新增「开发与接手」节（设计不变量、CI、测试说明），修正「五步→四步」措辞与 --work-dir 互斥说明

## [1.4.1]

- 单行模式 pending 上限冲刷（修无标点巨文件 O(n²)/OOM，P1）
- 编码检测采样边界容错
- 清理死代码、manifest 统一 UTF-8、work_dir 绝对化
- sort_file_inplace 并入能力探测体系
- 测试 75→81

## [1.4.0]

- 分桶右熵 k-way merge（省一趟全量 sort）
- 桶分发多进程分片（2→workers 进程，实测 1.7×）
- merge 阶段全流式、熵并行切分按字节均衡（消除热首字倾斜）
- 分桶路径端到端 + 续跑测试
- CLI 参数范围校验
- GitHub Actions CI（Linux/macOS/Windows）
- 测试 70→75

## [1.3.0]

- 断点续跑（--work-dir/--force）、logging + --verbose/--quiet
- 磁盘预检 + --temp-dir（sort -T 同步生效）
- 熵计算 O(1) 流式化、sort 能力探测 + 双排序内存均分
- 编码检测三采样
- 单行模式跨段词召回修复、混合输入丢数据等 P0 修复
- 阈值改显式传参
- 测试 57→70

## [1.2.0]

- 哈希分桶排序（300GB n-gram 2h→20min）
- 编码检测 v5 重写
- 编码杂质过滤
- 测试 38→57

## [1.1.0]

- 熵计算并行化
- GBK/GB18030 自动检测
- worker 直写 temp 文件、背压限流

## [1.0.0]

- 首版：多进程 N-gram、系统 sort 并行、词性标注、CLI 阈值修复
