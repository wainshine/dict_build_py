# HANDOVER

> 会话代际交接文档。每代会话接手时先读本文件，完成任务后更新「当前状态」与「欠账清单」。
> 本文件记"工作现场"；已发布的版本变更记 [CHANGELOG.md](CHANGELOG.md)；开发规范与不变量见 [README.md](README.md)「开发与接手」节。

最后更新：2026-09-03（四代会话）

## 当前状态

- 版本：v1.4.1，工作区干净，最后提交 2026-08-06（17f7138）
- 测试：81 passed（Python 3.14.2 本机实测；CI 覆盖 3.10/3.12/3.13 三平台）
- 文档：README.md + CHANGELOG.md + 本文件。项目规模小，不建 AGENTS.md，开发者接手信息在 README「开发与接手」节

## 四代会话审计结论（2026-09-03）

- 代码与 README 一致性：✅ 高度一致（13 个 CLI 参数、阈值、输出格式、分桶/背压/预检参数逐项核对）
- 功能 bug：未发现 P0/P1 级；checkpoint 恢复矩阵、分桶 fd 预算、单行模式边界拼接逻辑均自洽
- 测试匹配性：✅ 81 个测试与 1.4.1 功能一一对应

本次已完成：README 小瑕疵修正（五步→四步、--work-dir 互斥说明）、版本历史迁至 CHANGELOG.md、新建本文件。

## 欠账清单（按优先级）

| # | 事项 | 级别 | 说明 |
|---|------|------|------|
| 1 | 同步 dict-build skill 文档 | P2 | `~/.agents/skills/dict-build/SKILL.md` 停留在 v1.2.0 口径：缺 --work-dir/--temp-dir 等参数、位置概率公式误写为"×"（实为 min）、使用场景表数据与 README 不一致 |
| 2 | 补并行熵路径测试 | P2 | `_write_entropy_from_ngram_parallel` / `_split_sorted_ngram_by_chars`（>1GB 触发）无直接测试，可 monkeypatch `PARALLEL_ENTROPY_MIN_BYTES` 强制走该路径 |
| 3 | 补 BIG5 编码检测测试 | P2 | 现有编码测试仅覆盖 UTF-8/GBK |
| 4 | 补输出降序断言 | P3 | `extract_words` 按 freq 降序排序无直接断言 |
| 5 | pyproject classifiers 补 3.14 | P3 | 本机 3.14 实测通过，下个版本发布时顺手加上 |

## 低风险观察项（暂不需处理）

- `_detect_file_encoding` 中 `except (UnicodeDecodeError, LookupError)` 为不会触发的死分支（`errors="replace"` 不抛异常），无害
- 背压弹出 `futures.pop(0).get()` 若 worker 异常会传播中止（行为正确），但进度条未优雅关闭，仅影响显示

## 下一步候选任务

- 处理欠账 #1（同步 skill 文档，注意它在仓库外）
- 处理欠账 #2-4（一次提交可完成）
- 若有大规模语料实测需求，可验证分桶路径在真实 >1GB 场景的性能表现
