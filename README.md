# 低空经济通用大模型前沿论文库

这是一个独立的 GitHub Actions 项目，用于每天生成安卓手机浏览器可查看的前沿论文库。它不会影响已有的每日邮件提醒系统。

用户界面名称已升级为“低空经济通用大模型前沿论文库”；仓库名继续保留 `mobile-paper-library`，避免破坏 GitHub Pages 地址和既有 Secrets。

## 功能

- 每天北京时间 05:00 自动运行
- 最多检索 20 篇同时满足“低空场景 + LLM/VLM/VLA/World Model/时空基础模型/Agent/安全评估/边缘部署”的论文，不用临床、牙科、商业智能体或普通机器人论文凑数
- 从可下载 PDF 中递补选择 Top 10，单篇最大 25 MB、最多解析前 40 页
- Top 10 输出中英文标题与摘要、研究问题、方法链路、数据集、Baseline、Metrics、结果、局限和页码证据
- 每篇 Top 10 定位 1 张原始核心框图的 Figure/Page/Caption/裁剪坐标，并生成 1 张中文 Mermaid 总结框图
- PDF 仅临时下载；论文原图不提交到公开仓库，而是在个人 Obsidian 同步时本地重建
- 生成 GitHub Pages 手机网页归档
- 邮件包含执行摘要、Top 10 核心内容和其余论文简表，不再只发网页链接
- 保存 `outputs/YYYY-MM-DD.md` 与 `outputs/YYYY-MM-DD.json`，供 Obsidian 增量同步
- arXiv 限流或超时时采用退避重试；部分查询失败时保存 `partial` 日报

## GitHub Secrets

在新仓库 `Settings -> Secrets and variables -> Actions -> New repository secret` 添加：

| Name | Secret |
| --- | --- |
| `QQ_SMTP_HOST` | `smtp.qq.com` |
| `QQ_SMTP_PORT` | `465` |
| `QQ_SMTP_SSL` | `true` |
| `QQ_SMTP_USER` | `1425709546@qq.com` |
| `QQ_SMTP_FROM` | `1425709546@qq.com` |
| `QQ_SMTP_TO` | `1425709546@qq.com` |
| `QQ_SMTP_AUTH_CODE` | QQ 邮箱 SMTP 授权码 |
| `DEEPSEEK_API_KEY` | 优先使用的 DeepSeek API Key |
| `OPENAI_API_KEY` | DeepSeek 重试失败后的备用 OpenAI API Key |

`DEEPSEEK_API_KEY` 已配置时优先启用双语与全文结构化分析；可同时配置 `OPENAI_API_KEY` 作为失败重试后的备用模型。程序会先重试整批，再逐篇重试，并校验中文字段；仍失败的条目会标记为 `partial`，不会再错误显示为“免费模式”。没有配置模型时仍执行 PDF/核心图元数据核验，但不会虚构中文分析或数值结论。

## GitHub Pages

进入 `Settings -> Pages`：

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

保存后，网页地址通常是：

```text
https://smallopen123.github.io/mobile-paper-library/
```

## 手动测试

进入 `Actions -> Mobile Paper Library -> Run workflow`，手动运行一次。成功后会生成当天网页并发送一封单独邮件。

本地无邮件验证：

```powershell
python scripts/mobile_paper_library.py --dry-run --skip-llm
python -m unittest discover -s tests -v
```

历史重整不发送邮件。可在 GitHub Actions 的手动参数 `backfill_dates` 中填入
`2026-07-12,2026-07-13,...,2026-07-18`；流程会从 `sent_history.json` 与 arXiv 原始页面重建，
不根据旧标题列表臆造正文。
