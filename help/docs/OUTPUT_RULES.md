# 产出目录规范

从现在开始，本项目所有由代理生成的文档与统计产出统一放到 help 目录下。

- 文档（.md/.docx/.doc/说明文本）放在：help/docs/
- 指标汇总（.csv/汇总表）放在：help/metrics/
- 体检与检查报告（health_check、稳定性检查 JSON）放在：help/health_check/

补充约定：
- 若工具脚本默认输出到 work_dirs，应同步拷贝或迁移到 help 对应子目录。
- work_dirs 保留训练过程原始日志、权重与中间文件；对外分析材料以 help 目录为准。
