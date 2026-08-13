#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="overleaf"
OVERLEAF_BRANCH="master"

log() {
  echo
  echo "========== $1 =========="
}

run() {
  echo "+ $*"
  "$@"
}

trap 'echo; echo "❌ 脚本出错，停在第 $LINENO 行。请看上面的 Git 报错信息。"' ERR

log "检查 Git 仓库"
run git rev-parse --is-inside-work-tree >/dev/null

CURRENT_BRANCH="$(git branch --show-current || true)"
CURRENT_COMMIT="$(git rev-parse --short HEAD)"

echo "当前分支: ${CURRENT_BRANCH:-detached HEAD}"
echo "当前提交: $CURRENT_COMMIT"
echo "Overleaf remote: $REMOTE"
echo "Overleaf branch: $OVERLEAF_BRANCH"

log "检查 remote 是否存在"
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "❌ 找不到 remote: $REMOTE"
  echo "请先运行类似："
  echo "git remote add overleaf https://git@git.overleaf.com/你的项目ID"
  exit 1
fi

echo "Remote URL:"
git remote get-url "$REMOTE"

log "当前本地状态"
git status --short || true

log "暂存所有本地改动"
run git add --all

log "检查是否需要提交"
if ! git diff --cached --quiet; then
  COMMIT_MSG="agent: update $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "发现本地改动，准备提交：$COMMIT_MSG"
  run git commit -m "$COMMIT_MSG"
else
  echo "没有发现需要提交的本地改动。"
fi

log "从 Overleaf 获取网页端最新改动"
run git fetch "$REMOTE" "$OVERLEAF_BRANCH"

log "检查本地与 Overleaf 的差异"
AHEAD_BEHIND="$(git rev-list --left-right --count HEAD..."$REMOTE/$OVERLEAF_BRANCH" || echo "0 0")"
AHEAD="$(echo "$AHEAD_BEHIND" | awk '{print $1}')"
BEHIND="$(echo "$AHEAD_BEHIND" | awk '{print $2}')"

echo "本地领先 Overleaf: $AHEAD 个 commit"
echo "本地落后 Overleaf: $BEHIND 个 commit"

if [ "$BEHIND" -gt 0 ]; then
  log "Overleaf 网页端有新改动，开始 rebase"
  echo "如果这里发生 conflict，说明网页端和本地改到了同一部分，需要手动解决。"

  if ! git pull --rebase "$REMOTE" "$OVERLEAF_BRANCH"; then
    echo
    echo "❌ rebase 失败，通常是因为冲突。"
    echo
    echo "你可以这样处理："
    echo "1. 运行：git status"
    echo "2. 打开冲突文件，解决 <<<<<<< ======= >>>>>>> 标记"
    echo "3. 运行：git add 冲突文件"
    echo "4. 运行：git rebase --continue"
    echo "5. 再重新运行本脚本"
    echo
    echo "如果想放弃本次 rebase："
    echo "git rebase --abort"
    exit 1
  fi
else
  echo "Overleaf 网页端没有新的远程改动。"
fi

log "推送到 Overleaf"
run git push "$REMOTE" HEAD:"$OVERLEAF_BRANCH"

log "完成"
echo "✅ 已同步到 Overleaf。"
echo "你可以回到 Overleaf 网页端查看并重新编译。"
