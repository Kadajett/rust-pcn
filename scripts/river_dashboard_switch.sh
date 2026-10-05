#!/usr/bin/env bash
# Point the public River Song dashboard (portal + its helpers) at one telemetry directory.
#   scripts/river_dashboard_switch.sh b200      # the B200 run's mirror (fed by river-b200-bridge.service)
#   scripts/river_dashboard_switch.sh current   # the local run (default layout)
# Rewrites the --telemetry-dir / --publish-dir / --telemetry arguments and ReadWritePaths of the user units that read
# the telemetry dir, restarts them, and restarts the auditor (run under tmux, not a unit). The trainer services and the
# run guard are never touched: they keep writing to their own directories regardless of what the dashboard shows.
set -euo pipefail
RUNS=/bulk-storage/connectome-merc/river-multimodal-runs
case "${1:-}" in
    b200|current) TARGET=$RUNS/$1 ;;
    *) echo "usage: $0 b200|current" >&2; exit 2 ;;
esac
[ -d "$TARGET" ] || { echo "$TARGET does not exist" >&2; exit 1; }
UNITS=$HOME/.config/systemd/user
for unit in river-song-portal.service river-capability-eval.service river-live-prose-probe.service; do
    [ -f "$UNITS/$unit" ] || continue
    sed -i -E "s#$RUNS/(current|b200)#$TARGET#g" "$UNITS/$unit"
    if [ "$unit" = river-live-prose-probe.service ] && ! grep -q -- '--telemetry ' "$UNITS/$unit"; then
        sed -i -E "s#(live_prose_probe.py )#\1--telemetry $TARGET #" "$UNITS/$unit"
    fi
done
systemctl --user daemon-reload
systemctl --user restart river-song-portal.service
systemctl --user try-restart river-capability-eval.service river-live-prose-probe.service 2>/dev/null || true
# Auditor: a long-lived tmux process; replace it with one that reads the chosen directory.
pkill -f 'portal/auditor.py --telemetry-dir' || true
sleep 1
( cd "$HOME/Dev/rust-pcn" && nohup /usr/bin/python3 portal/auditor.py --telemetry-dir "$TARGET" \
    --audit-archive /bulk-storage/connectome-merc/river-quality-audits/audits.jsonl \
    --env-file "$HOME/.config/connectome-marty.env" --interval 30 --timeout 20 --sample-count 13 \
    > /tmp/river-auditor.log 2>&1 & )
sleep 2
echo "dashboard -> $TARGET"
systemctl --user is-active river-song-portal.service
pgrep -af 'portal/auditor.py' | cut -c1-120
curl -s -o /dev/null -w 'portal http %{http_code}\n' http://127.0.0.1:8799/
