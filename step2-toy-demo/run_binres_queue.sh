#!/bin/bash
# 二值残差流实验的无人值守队列:Claude 会话关掉以后,在服务器上把剩下的链跑完、评估、汇总并推到 GitHub。
#
# 循环做三件事,直到队列跑空、所有链结束:
#   1. 正在跑的链少于 MAXPAR 条时,按 QUEUE 的顺序启动还没启动过的链(./run_binres.sh <merge> <T> [seed] [wide]);
#   2. 一条链的 C 跑完后,做整句背对率评估(贪心与 K=4,不按长度归一化),写 results/improve/beam/<名字>.json;
#   3. 有新结果、或距上次提交超过 COMMIT_EVERY 秒时,重新生成汇总表(binres_tables.json / binres_tables.md),
#      把 results/improve 下的新文件(result.json、history、日志、model.pt、评估、汇总表)提交并推送到
#      分支 worktree-conversion-sweep 和 main(只做快进推送,失败只记日志,不强推)。
#
# 启动: ./start_binres_queue.sh(后台、脱离终端)
# 进度: cat results/improve/binres_queue_status.txt      日志: results/improve/binres_queue.log
# 停止: touch results/improve/binres_queue.stop  (不再启动新链,收尾后退出;已经在跑的链不受影响,要停它们用 kill)
# 试跑: DRY=1 ./run_binres_queue.sh once   只打印这一轮会做什么,不启动、不评估、不提交
#
# 一条链启动过(有 <名字>_chain.log)就不会再启动,即使它中途失败,免得反复重跑;要重跑先删掉那个日志。
cd "$(dirname "$0")" || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
OUT=results/improve
MAXPAR=${MAXPAR:-9}
POLL=${POLL:-120}
COMMIT_EVERY=${COMMIT_EVERY:-3600}
DRY=${DRY:-0}

# 第一批(2026-11-14 已启动,这里只负责收尾)
FIRST=("sr 1" "sr 2" "sr 4" "sr 8" "xor 1" "xor 2" "sat 4" "src 1" "src 2")
# 接着跑的,按优先级:src 的 T=4、8;三值 + 精确梯度的对照;加宽 d=256;src 的种子 1、2;xor 的 T=4、8
QUEUE=("src 4" "src 8"
       "satc 2" "satc 1" "satc 4"
       "src 1 0 wide" "src 2 0 wide" "src 4 0 wide"
       "src 2 1" "src 2 2" "src 4 1" "src 4 2" "src 8 1" "src 8 2"
       "xor 4" "xor 8")
# 汇总表里的对照:此前的三值残差链与实值残差基线
TERN_OLD="ternres_sat_T1,ternres_sat_T1_s1,ternres_sat_T1_s2,ternres_sat_T2,ternres_sat_T2_s1,ternres_sat_T2_s2"
BASELINES="T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k,T2=$OUT/T2_c,T2_s1=$OUT/T2_c_s1,T2_s2=$OUT/T2_c_s2,T4=$OUT/T4_c,T4_s1=$OUT/T4_c_s1,T4_s2=$OUT/T4_c_s2,T8=$OUT/T8_c,T1_W=$OUT/W_c,T4_W=$OUT/WT4_c"

log() { echo "[$(date '+%F %T')] $*"; }
name_of() { bash ./run_binres.sh name $1; }
n_running() { pgrep -fc "experiments/(binary|ternary)_residual.py chain" || true; }
is_running() { pgrep -f -- "chain --name $1 " > /dev/null; }
started() { [ -f "$OUT/$1_chain.log" ] || [ -f "$OUT/$1_c/result.json" ] || is_running "$1"; }
finished() { [ -f "$OUT/$1_c/result.json" ]; }

all_names() { local s; for s in "${FIRST[@]}" "${QUEUE[@]}"; do name_of "$s"; done; }

launch_next() {
  local s name
  [ -f "$OUT/binres_queue.stop" ] && return 0
  for s in "${QUEUE[@]}"; do
    [ "$(n_running)" -ge "$MAXPAR" ] && return 0
    name=$(name_of "$s")
    started "$name" && continue
    log "launch $name ($s)"
    if [ "$DRY" = 1 ]; then continue; fi
    bash ./run_binres.sh $s > /dev/null 2>&1 &
    sleep 20                                   # 等进程出现、日志文件建好,再数并行数
  done
}

evaluate_finished() {                          # 返回 0 表示有新的评估
  local name new=1
  for name in $(all_names); do
    finished "$name" || continue
    [ -f "$OUT/beam/$name.json" ] || [ -f "$OUT/beam/$name.failed" ] && continue
    log "evaluate $name"
    if [ "$DRY" = 1 ]; then continue; fi
    if $PY experiments/beam_eval.py --models "$name=$OUT/${name}_c/model.pt" --beams 1,4 --alphas 0 \
         --save "$OUT/beam/$name.json" > "$OUT/beam/$name.log" 2>&1; then
      new=0
    else
      log "evaluation of $name failed, see $OUT/beam/$name.log"
      touch "$OUT/beam/$name.failed"
    fi
  done
  return $new
}

make_tables() {
  local bin="" tern="$TERN_OLD" name beams
  for name in $(all_names); do
    case "$name" in binres_*) bin="$bin,$name";; *) tern="$tern,$name";; esac
  done
  beams=$(ls $OUT/beam/*.json 2> /dev/null | paste -sd, -)
  $PY experiments/binres_tables.py --runs "${bin#,}" --ternary "$tern" --baselines "$BASELINES" --beam "$beams" \
    --save "$OUT/binres_tables.json" > "$OUT/binres_tables.md" 2>> "$OUT/binres_queue.log" || log "binres_tables failed"
}

write_status() {
  local name state acc
  {
    echo "binres queue status, $(date '+%F %T')  (MAXPAR $MAXPAR; running chains: $(n_running))"
    for name in $(all_names); do
      if finished "$name"; then
        acc=$($PY -c "import json,sys; print('%.4f' % json.load(open(sys.argv[1]))['final']['token_acc'])" "$OUT/${name}_c/result.json" 2> /dev/null)
        state="done      C acc ${acc:-?}"
      elif is_running "$name"; then state="running"
      elif [ -f "$OUT/${name}_chain.log" ]; then state="STOPPED without a C result (see ${name}_chain.log)"
      else state="queued"
      fi
      printf '  %-26s %s\n' "$name" "$state"
    done
  } > "$OUT/binres_queue_status.txt"
}

commit_push() {                                # $1 = 提交说明
  [ "$DRY" = 1 ] && return 0
  git add -A "$OUT" > /dev/null 2>&1
  if git diff --cached --quiet; then return 0; fi
  git commit -q -F - <<EOF
binres queue: $1

Automatic commit by step2-toy-demo/run_binres_queue.sh (unattended binary-residual queue on GPU 1): run outputs under results/improve (result.json, history, chain logs, checkpoints), whole-sentence recall in results/improve/beam, and the regenerated binres_tables.json / binres_tables.md. Status at commit time is in results/improve/binres_queue_status.txt.

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WXSmKMbtt8Re7oceEtjWmJ
EOF
  if git push -q origin worktree-conversion-sweep worktree-conversion-sweep:main; then
    log "pushed: $1"
  else
    log "push failed (will retry with the next commit): $1"
  fi
}

last_commit=$(date +%s)
round() {                                      # 一轮;返回 0 表示全部结束
  local changed=1 now
  launch_next
  if evaluate_finished; then changed=0; fi
  write_status
  now=$(date +%s)
  if [ $changed = 0 ] || [ $((now - last_commit)) -ge "$COMMIT_EVERY" ]; then
    [ "$DRY" = 1 ] || make_tables
    if [ $changed = 0 ]; then commit_push "new results"; else commit_push "progress"; fi
    last_commit=$now
  fi
  # 结束条件:没有链在跑,而且队列里的都启动过了(或者收到停止标记)
  [ "$(n_running)" -gt 0 ] && return 1
  if [ ! -f "$OUT/binres_queue.stop" ]; then
    local s
    for s in "${QUEUE[@]}"; do started "$(name_of "$s")" || return 1; done
  fi
  return 0
}

log "queue started (pid $$, MAXPAR $MAXPAR, DRY $DRY)"
if [ "$1" = once ]; then round; write_status; cat "$OUT/binres_queue_status.txt"; exit 0; fi
until round; do sleep "$POLL"; done
evaluate_finished
write_status
[ "$DRY" = 1 ] || make_tables
echo "ALL DONE $(date '+%F %T')" >> "$OUT/binres_queue_status.txt"
commit_push "queue finished"
log "queue finished"
