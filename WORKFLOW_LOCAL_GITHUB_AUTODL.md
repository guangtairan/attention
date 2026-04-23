# Local-GitHub-AutoDL Workflow

This file records the standard sync workflow for this project:

- Local machine: edit code and validate.
- GitHub: central source of truth.
- AutoDL: pull latest code and run training.

## 1) Local machine -> GitHub

Run in local project root:

```bash
git status -sb
git add -A
git commit -m "your commit message"
git pull --rebase origin main
git push origin main
```

Notes:

- Always check `git status` before commit.
- If `pull --rebase` reports conflicts, resolve conflicts then:

```bash
git add <conflict_files>
git rebase --continue
```

Abort rebase if needed:

```bash
git rebase --abort
```

## 2) GitHub -> AutoDL

Run in AutoDL project root:

```bash
cd ~/autodl-tmp/mmsegmentation-sci
git status -sb
git fetch origin
git checkout main
git pull --rebase origin main
git log --oneline -n 3
```

If repository does not exist on AutoDL:

```bash
cd ~/autodl-tmp
git clone https://github.com/guangtairan/attention.git mmsegmentation-sci
cd mmsegmentation-sci
git checkout main
```

## 3) Typical AutoDL errors and fixes

### Error A

`fatal: not a git repository`

Reason: current directory is not the repo directory.

Fix:

```bash
cd ~/autodl-tmp/mmsegmentation-sci
```

### Error B

`cannot pull with rebase: You have unstaged changes`

Reason: local changes exist on AutoDL.

Fix option 1 (keep changes):

```bash
git stash push -u -m "autodl local changes"
git pull --rebase origin main
git stash pop
```

Fix option 2 (discard changes):

```bash
git restore .
git pull --rebase origin main
```

## 4) Branch policy for this project

- Main training branch: `main`.
- If a risky experiment needs isolation, create a feature branch:

```bash
git checkout -b exp/<name>
git push -u origin exp/<name>
```

Then pull the same branch on AutoDL before training.

## 5) Training reminder

- After pulling latest code on AutoDL, run training from the repo root.
- Keep `work_dir` unique per experiment/seed to avoid overwrite.
