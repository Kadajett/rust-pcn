#!/usr/bin/env bash
# Send River Song's training data and latest checkpoint to the rented B200, unpacking them at the same absolute paths
# as on this machine so the registry and checkpoint metadata need no rewrites. Waits for SSH, resumes partial uploads.
set -euo pipefail
# Replacement B200 of Oct 5 2026 (ubuntu@4.71.129.4 port 32317). The first instance (port 30204) was destroyed and may
# be reassigned: never target it. Our home IP is blocked at Hostinger's edge, so all traffic goes through the `b200`
# entry in ~/.ssh/config: ProxyJump via nucbox (Tailscale SSH), deploy key ~/.ssh/id_b200, shared control socket.
HOST=${HOST:-b200}
BUNDLE=/fast-storage/b200-bundle
ROOT=/bulk-storage/connectome-merc/river-universal-checkpoints/river-v8-fresh-seed20261004
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
now() { TZ=America/Los_Angeles date '+%-I:%M:%S %p'; }

until "${SSH[@]}" "$HOST" true 2>/dev/null; do sleep 10; done
echo "$(now) ssh up"
"${SSH[@]}" "$HOST" 'mkdir -p ~/b200-bundle && sudo mkdir -p /bulk-storage /fast-storage && sudo chown "$USER" /bulk-storage /fast-storage'

# Checkpoint: the active generation, manifest, health ledger and replay cache, frozen into a tarball now.
generation=$(python3 -c "import json;print(json.load(open('$ROOT/experts.json'))['generation'])")
echo "$(now) checkpoint generation $generation"
tar -C / -cf - "${ROOT#/}/experts.json" "${ROOT#/}/health.json" "${ROOT#/}/$generation" "${ROOT#/}/replay-cache" \
    | zstd -T4 -3 -q -f -o "$BUNDLE/river-checkpoint.tar.zst"

rsync_up() {
    rsync -a --partial --inplace --info=progress2 -e "ssh -o BatchMode=yes" "$@"
}
start=$(date +%s)
rsync_up "$BUNDLE/river-checkpoint.tar.zst" "$BUNDLE/river-data.tar.zst" "$BUNDLE/river-data.tar.zst.sha256" \
    "$HOST:b200-bundle/"
echo "$(now) upload done in $(( $(date +%s) - start )) s"
"${SSH[@]}" "$HOST" 'set -e; cd ~/b200-bundle && sha256sum -c river-data.tar.zst.sha256 \
    && zstd -dc river-data.tar.zst | tar -C / -xf - && zstd -dc river-checkpoint.tar.zst | tar -C / -xf - \
    && du -sh /bulk-storage/datasets/books /bulk-storage/localDocs/data/localDocs.db \
       /fast-storage/river-datasets/databricks-dolly-15k-v1 /bulk-storage/connectome-merc/*'
echo "$(now) unpacked on the B200"
