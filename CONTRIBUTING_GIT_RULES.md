# Git 提交规则（项目约定）

本项目为了保证仓库轻量、可复现，统一采用以下提交策略：

## 1) 永不提交的数据与训练产物

以下路径和文件只保留在本地，不上传到 GitHub：

- `data/`
- `work_dirs/`
- `*.pth`

## 2) 日常提交流程

每次提交前请先检查变更清单：

```bash
git status --short
git diff --name-only --cached
```

只提交代码和配置文件，不提交数据与训练输出。

## 3) 本地自动拦截

仓库内已提供本地 Git 钩子（`pre-commit`），会在 `git commit` 前自动拦截：

- `data/` 下文件
- `work_dirs/` 下文件
- `.pth` 权重文件

安装命令（Windows PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_git_hooks.ps1
```

安装后如误把上述文件加入暂存区，提交会被拒绝，并显示违规文件列表。
