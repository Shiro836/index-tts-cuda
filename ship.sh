#!/bin/bash
# ship.sh — one button: swap prod TTS from index-tts-updated (en/zh only) to the
# multilingual routing server (18 languages), with automatic rollback.
#
#   ./ship.sh              preflight, 10s abort window, then flip
#   ./ship.sh preflight    checks only — touches nothing, safe any time
#   ./ship.sh rollback     back to index-tts-updated
#   ./ship.sh promote      make multilingual the boot default (after a soak)
#   ./ship.sh status       what is serving right now
#
# Until the button is pressed, index-tts-updated keeps serving: this script is
# the only thing that stops it, and it puts it back itself if the new server
# fails to come up or fails an acceptance check.

set -uo pipefail

REPO=/home/forsen/repos/index-tts-ml
MODELS=/home/forsen/tts_models
NEW=index-tts-multilingual.service
OLD=index-tts-updated.service
PY=/home/forsen/miniconda3/envs/index-tts-vllm-fast/bin/python
URL=http://127.0.0.1:5113
ARTIFACTS=$REPO/ship_artifacts
BASELINE=$ARTIFACTS/baseline_en.json
HEALTH_TIMEOUT=900          # cold torch.compile boot takes minutes
VRAM_NEEDED_MB=15000        # measured 14.6GiB at --mel_workers 2

R="\033[31m"; G="\033[32m"; Y="\033[33m"; B="\033[1m"; N="\033[0m"
say()  { echo -e "${B}==>${N} $*"; }
ok()   { echo -e "  [${G}ok${N}] $*"; }
warn() { echo -e "  [${Y}warn${N}] $*"; }
bad()  { echo -e "  [${R}FAIL${N}] $*"; }

fails=0
check() { if [[ $1 == 0 ]]; then ok "$2"; else bad "$2 — $3"; fails=$((fails+1)); fi; }

# --- preflight: every check here runs against the untouched, still-serving box --

preflight() {
    say "Preflight"

    [[ -d $REPO ]] && [[ "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" == multilingual-routing ]]
    check $? "worktree $REPO is on branch multilingual-routing" "wrong branch"

    [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]
    check $? "worktree has no uncommitted changes" "commit or stash them first"

    local missing=0
    for d in base_pristine lora_step12000 lora_peft_step12000 glotlid; do
        [[ -e $MODELS/$d ]] || { missing=1; bad "missing weights: $MODELS/$d"; }
    done
    [[ $missing == 0 ]]
    check $? "weights present in $MODELS" "rsync them from the training machine"

    [[ -z "$(find "$MODELS" -xtype l 2>/dev/null | head -1)" ]]
    check $? "no dangling symlinks under $MODELS" \
          "re-run the transfer with rsync -aL (dereference)"

    local du_gb
    du_gb=$(du -s --block-size=1G "$MODELS" 2>/dev/null | cut -f1)
    [[ ${du_gb:-0} -ge 20 ]]
    check $? "weights are ~24GB on disk (${du_gb:-?}GB)" \
          "too small — a symlink farm was copied instead of the files"

    "$PY" -c "import fasttext, num2words" >/dev/null 2>&1
    check $? "fasttext-predict + num2words installed in the serving env" \
          "$PY -m pip install fasttext-predict num2words"

    ( cd "$REPO" && "$PY" -c "import indextts.lang_router, indextts.text_frontend_serve" ) >/dev/null 2>&1
    check $? "new server modules import cleanly" "see the traceback above"

    # gitignored build artifact: without it the server silently runs BigVGAN's
    # torch fallback, which costs 1.62s vs 0.29s per chunk under a 13-way burst
    [[ -f $REPO/indextts/BigVGAN/alias_free_activation/cuda/build/anti_alias_activation_cuda.so ]]
    check $? "prebuilt BigVGAN CUDA kernel present" \
          "cp it from /home/forsen/repos/index-tts-vllm-fast/indextts/BigVGAN/alias_free_activation/cuda/build/ (see SHIP.md)"

    [[ -f /etc/systemd/system/$NEW ]] && diff -q "$REPO/$NEW" /etc/systemd/system/$NEW >/dev/null
    check $? "$NEW installed and matches the repo copy" \
          "sudo cp $REPO/$NEW /etc/systemd/system/ && sudo systemctl daemon-reload"

    # Whichever TTS unit is up hands its VRAM back when the other one takes over,
    # so the budget is "free now + held by the incumbent" — and the incumbent is
    # the new unit once this has already been shipped.
    local serving=""
    systemctl is-active --quiet $NEW && serving=$NEW
    [[ -z $serving ]] && systemctl is-active --quiet $OLD && serving=$OLD

    local free_mb held_mb pid avail
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    held_mb=0
    if [[ -n $serving && -r /sys/fs/cgroup/system.slice/$serving/cgroup.procs ]]; then
        while read -r pid; do
            local used
            used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
                   | awk -F', *' -v p="$pid" '$1==p {print $2}')
            [[ -n ${used:-} ]] && held_mb=$((held_mb + used))
        done < /sys/fs/cgroup/system.slice/$serving/cgroup.procs
    fi
    avail=$((free_mb + held_mb))
    [[ $avail -ge $VRAM_NEEDED_MB ]]
    check $? "VRAM after the swap: ${avail}MiB (${free_mb} free + ${held_mb} released by ${serving:-nothing}), need ${VRAM_NEEDED_MB}MiB" \
          "stop another GPU tenant first (ollama llama-server is the big one)"

    if [[ $serving == $NEW ]]; then
        warn "$NEW is ALREADY running — nothing to flip"
    elif [[ $serving == $OLD ]]; then
        ok "$OLD is serving (rollback target confirmed)"
    else
        bad "neither TTS unit is active — refusing to flip into an unknown state"
        fails=$((fails+1))
    fi

    if [[ $fails -gt 0 ]]; then
        echo -e "\n${R}Preflight failed ($fails).${N} Nothing was changed; ${serving:-no TTS unit} is still serving."
        return 1
    fi
    echo -e "\n${G}Preflight clean.${N} ${serving:-No TTS unit} is still serving — nothing has changed yet."
    return 0
}

# Stop a unit and wait for its EngineCore children to actually hand the VRAM
# back. Do NOT lean on the units' Conflicts= for this: the reverse edge only
# exists while both units are loaded (an inactive unit is loaded lazily, and
# `systemctl show index-tts-updated -p ConflictedBy` is empty on this box), and
# even when it does fire, systemd gives no ordering guarantee against the new
# unit's ExecStartPre=wait_for_vram — the new server could start waiting for
# VRAM the old one has not released yet. Conflicts= stays in the unit as a
# belt-and-braces guard against both units being started at boot.
stop_and_wait_for_vram() {
    local unit=$1 deadline=$((SECONDS + 120)) free_mb
    say "Stopping $unit and waiting for the VRAM"
    sudo systemctl stop "$unit"
    while (( SECONDS < deadline )); do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [[ ${free_mb:-0} -ge $VRAM_NEEDED_MB ]] && ! ss -ltn 2>/dev/null | grep -q ':5113 '; then
            ok "${free_mb}MiB free, port 5113 released"
            return 0
        fi
        sleep 3
    done
    warn "after 120s: ${free_mb:-?}MiB free (need ${VRAM_NEEDED_MB}MiB), port 5113 $(ss -ltn 2>/dev/null | grep -q ':5113 ' && echo BUSY || echo free)"
    warn "starting anyway — the unit's own ExecStartPre=wait_for_vram will keep waiting"
    return 0
}

wait_healthy() {
    local unit=$1 deadline=$((SECONDS + HEALTH_TIMEOUT))
    while (( SECONDS < deadline )); do
        if ! systemctl is-active --quiet "$unit"; then
            local state
            state=$(systemctl is-active "$unit")
            if [[ $state == failed ]]; then
                bad "$unit entered failed state"
                return 1
            fi
        fi
        if curl -fsS --max-time 5 "$URL/health" 2>/dev/null | grep -q '"status": *"healthy"'; then
            ok "$unit healthy after ${SECONDS}s"
            return 0
        fi
        sleep 5
    done
    bad "$unit did not become healthy within ${HEALTH_TIMEOUT}s"
    return 1
}

rollback() {
    say "ROLLING BACK to $OLD"
    stop_and_wait_for_vram $NEW
    sudo systemctl start $OLD || true
    if wait_healthy $OLD; then
        echo -e "${G}Rolled back.${N} Prod is serving the old (en/zh) model again."
    else
        echo -e "${R}ROLLBACK DID NOT COME UP.${N} TTS is DOWN. Look at:"
        echo "  journalctl -u $OLD -n 100 --no-pager"
        echo "  nvidia-smi   # orphan EngineCore processes hold VRAM and OOM the next start"
    fi
    orphan_check
}

orphan_check() {
    local orphans
    orphans=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader \
              | grep -i "EngineCore" | wc -l)
    echo "  EngineCore processes on the GPU: $orphans"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/    /'
}

ship() {
    preflight || exit 1

    mkdir -p "$ARTIFACTS"
    say "Recording the English baseline from the live server (for post-flip comparison)"
    "$PY" "$REPO/tools/ship_checks.py" baseline --url "$URL" --out "$BASELINE" \
        --artifacts "$ARTIFACTS"

    if [[ ${1:-} != "-y" ]]; then
        echo
        echo -e "${Y}About to stop $OLD and start $NEW on port 5113.${N}"
        echo "TTS will be unavailable for the few minutes the new server takes to boot."
        echo -n "Ctrl-C to abort: "
        for i in 10 9 8 7 6 5 4 3 2 1; do echo -n "$i "; sleep 1; done
        echo
    fi

    say "Flipping"
    stop_and_wait_for_vram $OLD
    if ! sudo systemctl start $NEW; then
        bad "systemctl start $NEW failed"
        rollback
        exit 1
    fi

    if ! wait_healthy $NEW; then
        echo "  last 40 log lines:"
        journalctl -u $NEW -n 40 --no-pager | sed 's/^/    /'
        rollback
        exit 1
    fi

    say "Acceptance checks"
    if ! "$PY" "$REPO/tools/ship_checks.py" smoke --url "$URL" --baseline "$BASELINE" \
            --artifacts "$ARTIFACTS"; then
        bad "acceptance checks failed"
        rollback
        exit 1
    fi

    cat <<EOF

$(echo -e "${G}${B}SHIPPED.${N}") The multilingual server is serving prod on port 5113.

Still to do, by you:
  1. LISTEN to $ARTIFACTS/ — English (baseline_en.wav vs new_en_forced.wav) must be
     indistinguishable, and Russian must speak "500" as Russian words. Nobody has
     ear-checked ba/tt/be/ja at all.
  2. Watch it under real traffic for a while:
       journalctl -u $NEW -f
  3. This is a RUNTIME flip only — a reboot still brings back the old server.
     After the soak, make it the boot default:
       $REPO/ship.sh promote

Rollback at any time (one command, no preflight, ~1 min):
  $REPO/ship.sh rollback
EOF
}

promote() {
    systemctl is-active --quiet $NEW || {
        bad "$NEW is not running — refusing to make a non-serving unit the boot default"
        exit 1
    }
    say "Making $NEW the boot default"
    sudo systemctl enable $NEW && sudo systemctl disable $OLD
    echo -e "${G}Done.${N} $NEW now starts at boot; $OLD stays installed for rollback."
}

status() {
    for u in $NEW $OLD index-tts.service; do
        printf '  %-34s %-10s %s\n' "$u" "$(systemctl is-active $u)" "$(systemctl is-enabled $u 2>/dev/null)"
    done
    echo
    curl -fsS --max-time 5 "$URL/health" 2>/dev/null | "$PY" -m json.tool 2>/dev/null \
        || echo "  no healthy server on $URL"
    echo
    orphan_check
}

case "${1:-ship}" in
    ship)      ship "${2:-}" ;;
    -y)        ship -y ;;
    preflight|check) preflight ;;
    rollback)  rollback ;;
    promote)   promote ;;
    status)    status ;;
    *) echo "usage: $0 [ship [-y] | preflight | rollback | promote | status]"; exit 2 ;;
esac
