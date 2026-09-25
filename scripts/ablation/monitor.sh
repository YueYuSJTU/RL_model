#!/bin/bash
# ============================================================================
# 消融实验监控：池名册 + battle cycle 进度 + 关键日志行 + 池更新决策统计
#
# 用法：bash scripts/ablation/monitor.sh <run_dir> [tail_lines]
# 例：  bash scripts/ablation/monitor.sh experiments/20260924_225859_selfplay_init
#
# 说明：stage 级 logger 用的是 SB3 configure(csv/tensorboard)，其 logger.info 不会
#       打到 stdout，因此 "BATTLE-TRAINING Cycle N" 横幅不会出现在日志里；本脚本以
#       (a) 池名册、(b) cycle 目录数、(c) stdout 日志中的 logging.* 行 三者为准。
# ============================================================================
set -uo pipefail
cd /home/ubuntu/Workfile/RL/RL_model

RUN=${1:?用法: monitor.sh <run_dir> [tail_lines]}
TAIL=${2:-5}

if [ ! -d "$RUN" ]; then
    echo "!! 目录不存在: $RUN"
    exit 1
fi
BASE=$(basename "$RUN")

echo "==================================================================="
echo "RUN: $RUN"
echo "现在: $(date '+%F %T')"
echo "==================================================================="

echo
echo "--- 1) 对手池名册（编号 / md5 / 大小）--------------------------------"
if [ -d "$RUN/opponent_pool" ]; then
    for d in $(ls -d "$RUN"/opponent_pool/*/ 2>/dev/null | sort -V); do
        n=$(basename "$d")
        if [ -f "$d/best_model.zip" ]; then
            printf "  %-4s %s  %10s bytes\n" "$n" \
                "$(md5sum "$d/best_model.zip" | awk '{print $1}')" \
                "$(stat -c %s "$d/best_model.zip")"
        else
            printf "  %-4s (缺少 best_model.zip)\n" "$n"
        fi
    done
    echo "  池内模型数: $(ls -d "$RUN"/opponent_pool/*/ 2>/dev/null | wc -l)"
else
    echo "  (尚无 opponent_pool 目录)"
fi

echo
echo "--- 2) cycle 进度 ----------------------------------------------------"
for st in stage1 stage2; do
    [ -d "$RUN/$st" ] || continue
    cnt=$(ls -d "$RUN/$st"/*cycle* 2>/dev/null | wc -l)
    last=$(ls -d "$RUN/$st"/*cycle* 2>/dev/null | tail -1 | xargs -r basename)
    echo "  $st: ${cnt} cycles（最新: ${last:-无}）"
done

echo
echo "--- 3) 池更新事件（stdout 日志）-------------------------------------"
LOG=""
DERIVED=$(echo "$BASE" | sed "s/^[0-9]\{8\}_[0-9]\{6\}_//")
for cand in "logs/ablation/${DERIVED}.log" "logs/ablation/${BASE}.log" "logs/ablation/${BASE}_battle.log"; do
    [ -f "$cand" ] && LOG="$cand" && break
done
if [ -n "$LOG" ]; then
    echo "  日志: $LOG"
    grep -n "\[opponent_pool\]\|NNVecEnv initialized\|白名单\|完整复制到\|正在用新模型替换\|对手池已满\|未能战胜\|Traceback\|Error" "$LOG" | tail -"$TAIL"
    echo "  ----"
    echo "  无条件插入(未满->添加): $(grep -c "对手池未满" "$LOG" 2>/dev/null)"
    echo "  真实替换(新模型替换)  : $(grep -c "正在用新模型替换" "$LOG" 2>/dev/null)"
    echo "  挑战未替换(未能战胜)  : $(grep -c "未能战胜" "$LOG" 2>/dev/null)"
else
    echo "  (未找到 stdout 日志，请确认 tee 的目标路径)"
fi

echo
echo "--- 4) 训练曲线文件 --------------------------------------------------"
for st in stage2 stage1; do
    [ -d "$RUN/$st/metrics" ] || continue
    echo "  $st/metrics: $(ls "$RUN/$st/metrics" 2>/dev/null | tr '\n' ' ')"
    [ -f "$RUN/$st/metrics/progress.csv" ] && echo "    progress.csv 大小: $(stat -c %s "$RUN/$st/metrics/progress.csv") bytes"
done
echo "  每 cycle 曲线:"
ls -la "$RUN"/tensorboard/battle/PPO_0/ 2>/dev/null | awk 'NR>3 {print "    ", $5, $9}'
echo "  最新 cycle 的 evaluations.npz:"
ls -la "$RUN"/stage2/*cycle*/evaluations.npz 2>/dev/null | tail -2 | awk '{print "    ", $5, $9}'
echo "  课程进度(stdout 日志最后一条):"
grep "课程更新" "$LOG" 2>/dev/null | tail -1
echo
