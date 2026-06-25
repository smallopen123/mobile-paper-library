# 低空经济前沿双语论文库

这是一个独立的 GitHub Actions 项目，用于每天生成安卓手机浏览器可查看的前沿论文库。它不会影响已有的每日邮件提醒系统。

当前版本为无 API 免费模式，不调用 OpenAI 或其他模型 API。

## 功能

- 每天北京时间 09:00 自动运行
- 检索 20 篇低空经济安全、航迹预测、智能体、机器人学习相关论文
- 生成 GitHub Pages 手机网页归档
- 每篇包含原文页面、PDF 链接、英文摘要、规则相关性说明、实践阅读思路
- 中文翻译建议使用安卓 Chrome/Edge 的网页翻译功能
- 单独发送邮件通知，邮件中包含当天网页链接

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

无需配置 `OPENAI_API_KEY`。

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
