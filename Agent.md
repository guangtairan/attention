# 本地-GitHub-AutoDL 协作流程

本文档用于统一本项目的代码同步、训练执行与编码规范。后续所有项目协作默认以本文件为准。

- 本地机器：修改代码并先本地验证。
- GitHub：作为唯一代码事实源。
- AutoDL：拉取最新代码并执行训练。

## 1）本地机器 -> GitHub

在本地项目根目录执行：

```bash
git status -sb
git add -A
git commit -m "提交说明"
git pull --rebase origin main
git push origin main
```

说明：

- 每次提交前先检查 `git status`。
- 如果 `pull --rebase` 出现冲突，先解决冲突再执行：

```bash
git add <冲突文件>
git rebase --continue
```

如需中止 rebase：

```bash
git rebase --abort
```

## 2）GitHub -> AutoDL

在 AutoDL 项目根目录执行：

```bash
cd ~/autodl-tmp/mmsegmentation-sci
git status -sb
git fetch origin
git checkout main
git pull --rebase origin main
git log --oneline -n 3
```

如果 AutoDL 上还没有仓库：

```bash
cd ~/autodl-tmp
git clone https://github.com/guangtairan/attention.git mmsegmentation-sci
cd mmsegmentation-sci
git checkout main
```

## 3）AutoDL 常见报错与处理

### 报错 A

`fatal: not a git repository`

原因：当前目录不是仓库目录。

处理：

```bash
cd ~/autodl-tmp/mmsegmentation-sci
```

### 报错 B

`cannot pull with rebase: You have unstaged changes`

原因：AutoDL 本地有未提交修改。

处理方案 1（保留修改）：

```bash
git stash push -u -m "autodl local changes"
git pull --rebase origin main
git stash pop
```

处理方案 2（丢弃修改）：

```bash
git restore .
git pull --rebase origin main
```

## 4）分支策略

- 默认训练分支：`main`。
- 需要隔离高风险实验时，新建实验分支：

```bash
git checkout -b exp/<name>
git push -u origin exp/<name>
```

然后在 AutoDL 拉取同名分支再训练。

## 5）训练提醒

- AutoDL 拉取最新代码后，必须在仓库根目录运行训练。
- 每个实验/随机种子使用独立 `work_dir`，避免相互覆盖。

## 6）编码与隐藏字符安全规范（重点）

- 所有项目 `.py` 文件统一保存为 UTF-8 无 BOM。
- 避免在代码/配置中出现以下隐藏字符：
- BOM（`\ufeff`）
- 零宽字符（`\u200b`、`\u200c`、`\u200d`、`\u2060`）
- 不间断空格（`\xa0`）

当出现如下配置解析报错时：

- `SyntaxError: invalid character in identifier`
- 或报错定位在 `ast.parse(...)` 且指向配置文件首行注释（例如 `# Unified optimizer config for paper experiments`）

优先按下面流程处理：

```bash
python scripts/clean_configs_hidden_chars.py --root configs
python -m py_compile configs/_base_/schedules/schedule_20k.py
```

必要时可再做字节级确认（检查是否以 `EF BB BF` 开头）。