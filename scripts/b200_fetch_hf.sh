#!/usr/bin/env bash
# Run on the B200, detached by the caller: nohup bash b200_fetch_hf.sh > ~/hf-fetch.log 2>&1 &
# All caches, temporary files and helpers stay inside the data task's approved roots.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REGISTRY=${1:-"$SCRIPT_DIR/../datasets/training-registry-prose.json"}
export HF_HOME="$HOME/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$HF_HOME/xdg"
export TMPDIR="$HF_HOME/tmp"
export PIP_CACHE_DIR="$HF_HOME/pip"
export PYTHONPYCACHEPREFIX="$HF_HOME/pycache"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TZ=America/Los_Angeles
WORK="$HF_HOME/b200-fetch"
VENV="$HOME/venv/b200-hf"
mkdir -p "$TMPDIR" "$WORK/results" "$HOME/venv"
exec 9>"$WORK/fetch.lock"
flock -n 9 || { echo 'A B200 fetch is already running'; exit 1; }
printf 'Starting B200 fetch at %s\n' "$(date '+%Y-%m-%d %-I:%M:%S %p %Z')"
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv --without-pip "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version; then
    if ! "$VENV/bin/python" -m ensurepip; then
        curl --fail --location --silent --show-error https://bootstrap.pypa.io/get-pip.py -o "$WORK/get-pip.py"
        "$VENV/bin/python" "$WORK/get-pip.py"
    fi
fi
"$VENV/bin/python" -m pip install -U huggingface_hub datasets pyarrow
if "$VENV/bin/python" -m pip install -U hf_transfer; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export PATH="$VENV/bin:$PATH"
python -c 'import datasets, huggingface_hub, pyarrow; print("Versions:", datasets.__version__, huggingface_hub.__version__, pyarrow.__version__)'
for directory in /bulk-storage/datasets/hf-text /fast-storage/river-datasets; do
    if [[ ! -d "$directory" ]]; then
        mkdir -p "$directory" || sudo -n install -d -o "$(id -u)" -g "$(id -g)" "$directory"
    fi
done
pids=()
ids=()
for id in tinystories-v1 fineweb-edu-sample-10bt-v1 wikipedia-20231101-en-v1; do
    python "$SCRIPT_DIR/b200_fetch_text.py" --id "$id" --results-dir "$WORK/results" &
    pids+=("$!")
    ids+=("$id")
done
for id in openassistant-oasst2-v1 rajpurkar-squad-v2-v1 hotpotqa-distractor-v1 openai-gsm8k-main-v1; do
    python "$SCRIPT_DIR/b200_prepare_hf.py" --id "$id" --registry "$REGISTRY" --results-dir "$WORK/results" &
    pids+=("$!")
    ids+=("$id")
done
failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'Completed %s\n' "${ids[$index]}"
    else
        printf 'FAILED %s\n' "${ids[$index]}"
        failed=1
    fi
done
python "$SCRIPT_DIR/b200_finalize_registry.py" --base "$REGISTRY" --results-dir "$WORK/results" --output "$WORK/training-registry-b200.json"
printf 'Finished B200 fetch at %s; failures=%s\n' "$(date '+%Y-%m-%d %-I:%M:%S %p %Z')" "$failed"
exit "$failed"
