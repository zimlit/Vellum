#!/usr/bin/env bash
# Switch away from vellum's VT and back, repeatedly, and check it survives.
#
# Run from SSH with sudo while vellum runs on its VT:
#
#   sudo scripts/vt-cycle.sh [-n ROUNDS] [-s SECONDS] [-v OUR_VT] [-o OTHER_VT] [-l LOG]
#
#   -n  round trips (default 20)
#   -s  pause after each switch, in seconds (default 0.5)
#   -v  vellum's VT (default: the foreground VT when the script starts)
#   -o  VT to switch to (default: 1, or 2 if vellum is on 1)
#   -l  vellum's log (default: ~/.local/state/vellum/vellum.log of the
#       user who ran sudo, or $XDG_STATE_HOME/vellum/vellum.log)
#
# Each round must finish switching within 5 s each way (chvt waits for the
# switch, which waits for vellum to let go of the display), and vellum must
# still be running afterwards. At the end, the log must show one "session
# disabled" and one "session enabled" line per round.

set -euo pipefail

rounds=20
pause=0.5
ours=""
other=""
log=""

while getopts "n:s:v:o:l:h" opt; do
    case $opt in
        n) rounds=$OPTARG ;;
        s) pause=$OPTARG ;;
        v) ours=$OPTARG ;;
        o) other=$OPTARG ;;
        l) log=$OPTARG ;;
        *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "needs root for chvt: run with sudo" >&2
    exit 2
fi

ours=${ours:-$(fgconsole)}
if [[ -z $other ]]; then
    other=$([[ $ours == 1 ]] && echo 2 || echo 1)
fi
if [[ -z $log ]]; then
    user=${SUDO_USER:-$USER}
    home=$(getent passwd "$user" | cut -d: -f6)
    log=${XDG_STATE_HOME:-$home/.local/state}/vellum/vellum.log
fi

# The front end, not the session server (same binary name).
vellum_pid() {
    local pid
    for pid in $(pgrep -x vellum || true); do
        if ! tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q -- --session-server; then
            echo "$pid"
            return
        fi
    done
}

pid=$(vellum_pid)
if [[ -z $pid ]]; then
    echo "vellum isn't running" >&2
    exit 1
fi
if [[ ! -r $log ]]; then
    echo "can't read the log at $log (use -l)" >&2
    exit 1
fi
start_line=$(wc -l <"$log")

echo "vellum (pid $pid) on VT $ours; $rounds round trips to VT $other, ${pause}s pauses"
echo "log: $log"

switch() {
    if ! timeout 5 chvt "$1"; then
        echo "FAIL: switching to VT $1 didn't finish within 5 s" >&2
        exit 1
    fi
    sleep "$pause"
}

for ((i = 1; i <= rounds; i++)); do
    switch "$other"
    switch "$ours"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "FAIL: vellum exited during round $i" >&2
        tail -n 20 "$log" >&2
        exit 1
    fi
    printf '\rround %d/%d ok' "$i" "$rounds"
done
echo

# Give the last enable a moment to reach the log.
sleep 0.5
new=$(tail -n "+$((start_line + 1))" "$log")
disabled=$(grep -c "session disabled" <<<"$new" || true)
enabled=$(grep -c "session enabled" <<<"$new" || true)
echo "log: $disabled disabled, $enabled enabled"
if [[ $disabled -ne $rounds || $enabled -ne $rounds ]]; then
    echo "FAIL: expected $rounds of each" >&2
    exit 1
fi
errors=$(grep -cE " (ERROR|WARN) " <<<"$new" || true)
if [[ $errors -gt 0 ]]; then
    echo "note: $errors warning/error lines in the log since the start:"
    grep -E " (ERROR|WARN) " <<<"$new" | head -n 10
fi
echo "PASS"
