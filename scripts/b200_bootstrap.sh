#!/usr/bin/env bash
# b200_bootstrap.sh — idempotent setup of a rented Ubuntu 24.04 + NVIDIA B200 (sm_100) box for River Song.
#
# What it does (each step is guarded, re-running is a no-op; every step prints `[bootstrap] <step>: done|skipped`):
#   1. apt packages the cargo build needs (build-essential pkg-config libssl-dev libsqlite3-dev zstd rsync git curl ...)
#   2. data paths, EARLY so a transfer bundle unpacked later lands on the big disk: the same absolute paths as the dev
#      box (/bulk-storage/..., /fast-storage/...), symlinked into the largest data disk when that is not the root
#      filesystem. If /bulk-storage or /fast-storage already exist as real directories on a smaller filesystem (the
#      transfer bundle ran first) it prints a loud WARNING with sizes and the exact move commands; it only moves them
#      with MOVE_DATA_DIRS=1. Warns when the data disk has < 60 GB free.
#   3. NVIDIA driver check via nvidia-smi (fails loudly if absent; never installs drivers); records the driver's
#      'CUDA Version: X.Y' banner
#   4. CUDA toolkit for NVRTC: kernels go NVRTC -> PTX -> driver JIT, so the toolkit must be >= 12.8 (sm_100) AND
#      <= the driver's CUDA version (newer PTX fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Picks the highest
#      installed toolkit in that window, prints every toolkit found with its verdict, installs cuda-toolkit-12-8 from the
#      NVIDIA apt repo when none qualifies. Then checks the NVRTC runtime headers in $CUDA_HOME/include (cuda_fp16.h
#      cuda_bf16.h mma.h cuda_runtime.h crt/) and installs cuda-nvrtc-dev/cudart-dev/crt/cccl-<v> if any is missing.
#      Checks that an UNVERSIONED libcuda.so (dlopen name used by cudarc 0.12.1) and libnvrtc.so exist; symlinks them
#      with sudo, or into ~/.river-cuda-compat (prepended to LD_LIBRARY_PATH) without sudo. Writes
#      /etc/profile.d/river-cuda.sh (or ~/.river-b200.env without sudo) exporting CUDA_PATH, CUDA_HOME, PATH,
#      LD_LIBRARY_PATH, JEV_PCN_CUDA_LIB_DIR
#   5. rustup stable (minimal profile)
#   6. clone https://github.com/Kadajett/rust-pcn into /home/kadajett/Dev/rust-pcn (same absolute path as the dev box)
#   7. cargo build --locked --release --features cuda, only the required targets (train/widen/repair + gpu_smoke; BUILD_ALL=1
#      builds --bins) with all cores; copy the binaries to target/live/ (deployed exe is separate from build outputs)
#   8. GPU smoke: nvidia-smi, trainer --help, and a 1024x1024 cubecl matmul on device 0 (examples/gpu_smoke.rs) with
#      CUBECL_DEBUG_LOG=/tmp/cubecl-smoke.log; reports (not gates) whether a wmma (tensor-core) kernel compiled
#   9. summary: driver CUDA, CUDA_HOME + libnvrtc + headers, libcuda.so status, data disk + free space, commit, hashes, smoke
#
# Usage (from the dev box). Replacement B200 of Oct 5 2026: ubuntu@4.71.129.4 port 32317, password-only (the first
# instance on port 30204 was destroyed; never contact it; nothing survives from it, so run this from scratch):
#   ssh-copy-id -p 32317 ubuntu@4.71.129.4            # once, interactive password; makes every later step unattended
#   scp -P 32317 scripts/b200_bootstrap.sh ubuntu@4.71.129.4:
#   ssh -p 32317 ubuntu@4.71.129.4 'bash b200_bootstrap.sh'
# Run as the login user with passwordless sudo (root works too). Re-run freely after fixing anything.
#
# Environment overrides:
#   CUDA_TOOLKIT_PKG   apt package to install when no toolkit in [12.8, driver CUDA] is found (default cuda-toolkit-12-8;
#                      the resolved home is then /usr/local/cuda-12.8)
#   CUDA_HOME_OVERRIDE force this CUDA home (skips toolkit discovery/installation; header/lib checks still run)
#   DATA_DISK          mount point of the big data disk (default: auto-detect the mounted filesystem with the most free
#                      space; if that is /, real directories are created on the root filesystem)
#   MOVE_DATA_DIRS=1   move pre-existing real /bulk-storage and /fast-storage directories onto the data disk (default:
#                      warn and print the commands only)
#   BUILD_ALL=1        cargo build --bins instead of only the required binaries
#   REPO_REF           git ref/commit to check out (default: main, fast-forwarded to origin)
#   SKIP_BUILD=1       skip the cargo build (step 7)
#   SKIP_SMOKE=1       skip the GPU smoke (step 8)
set -euo pipefail

REPO_URL=https://github.com/Kadajett/rust-pcn
USER_HOME=/home/kadajett
REPO_DIR=$USER_HOME/Dev/rust-pcn
LIVE_DIR=$REPO_DIR/target/live
CUDA_TOOLKIT_PKG=${CUDA_TOOLKIT_PKG:-cuda-toolkit-12-8}
CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
MIN_CUDA_MAJOR=12
MIN_CUDA_MINOR=8
APT_PACKAGES=(build-essential pkg-config libssl-dev libsqlite3-dev zstd rsync git curl ca-certificates python3)
BINARIES=(river-pcn-train-universal river-pcn-widen-universal river-pcn-repair-universal)
BUILD_ALL_BINARIES=(river-pcn-train-universal river-pcn-widen-universal river-pcn-expand-universal-experts
          river-pcn-repair-universal river-pcn-migrate-universal river-pcn-generate river-pcn-eval-multimodal)
NVRTC_HEADERS=(cuda_fp16.h cuda_bf16.h mma.h cuda_runtime.h crt/)
MIN_DATA_FREE_GB=60
# Absolute data paths referenced by datasets/training-registry-prose.json and the run plan (parents of files).
DATA_DIRS=(
    /bulk-storage/datasets/books
    /bulk-storage/localDocs/data
    /bulk-storage/datasets/images/tinyimagenet-converted
    /bulk-storage/datasets/images/cifar100-converted
    /bulk-storage/datasets/images/svhn-converted
    /bulk-storage/datasets/images/fashion-mnist-converted
    /bulk-storage/datasets/images/flowers102-converted
    /bulk-storage/datasets/images/stl10-converted
    /bulk-storage/connectome-merc/marty-continuous-20260916
    /bulk-storage/connectome-merc/river-universal-checkpoints
    /bulk-storage/connectome-merc/river-multimodal-runs
    /fast-storage/river-datasets/databricks-dolly-15k-v1
    /fast-storage/river-datasets/zefancai-open-jev-release-v2-v1
)

LOGIN_USER=${SUDO_USER:-$(id -un)}
if [ "$(id -u)" -eq 0 ]; then
    HAVE_SUDO=1
    as_root() { "$@"; }
elif sudo -n true 2>/dev/null; then
    HAVE_SUDO=1
    as_root() { sudo -n "$@"; }
elif [ -t 0 ] && sudo -v; then
    HAVE_SUDO=1
    as_root() { sudo "$@"; }
else
    HAVE_SUDO=0
    as_root() { echo "[bootstrap] no sudo: cannot run: $*" >&2; return 1; }
fi

SUMMARY=()
log() { echo "[bootstrap] $*"; }
done_step() { log "$1: done${2:+ ($2)}"; SUMMARY+=("$1: done${2:+ ($2)}"); }
skip_step() { log "$1: skipped ($2)"; SUMMARY+=("$1: skipped ($2)"); }
fail() { echo "[bootstrap] FAILED: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. apt
step_apt() {
    local missing=()
    local pkg
    for pkg in "${APT_PACKAGES[@]}"; do
        if [ "$(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null || true)" != "install ok installed" ]; then
            missing+=("$pkg")
        fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        skip_step apt "all ${#APT_PACKAGES[@]} packages installed"
        return
    fi
    [ "$HAVE_SUDO" -eq 1 ] || fail "apt: missing ${missing[*]} and no sudo to install them"
    log "apt: installing ${missing[*]}"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -qq update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq install "${missing[@]}"
    done_step apt "installed ${missing[*]}"
}

# ---------------------------------------------------------------- 3. driver
DRIVER_VERSION=""
DRIVER_CUDA=""      # 'CUDA Version: X.Y' from the nvidia-smi banner = highest PTX version the driver can JIT
step_driver() {
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not on PATH: no NVIDIA driver on this image (not installing drivers)"
    local smi
    smi=$(nvidia-smi 2>&1) || fail "nvidia-smi failed: $smi"
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')
    DRIVER_CUDA=$(printf '%s\n' "$smi" | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}')
    local gpu_name compute_cap
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)
    log "driver: $gpu_name, driver $DRIVER_VERSION, driver CUDA $DRIVER_CUDA, compute capability ${compute_cap:-?}"
    version_ge "${DRIVER_CUDA:-0}" "$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR" \
        || fail "driver reports CUDA $DRIVER_CUDA < $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR; sm_100 kernels from NVRTC 12.8 need a newer driver (not installing drivers)"
    done_step driver "$gpu_name driver $DRIVER_VERSION / CUDA $DRIVER_CUDA"
}

# ---------------------------------------------------------------- 4. cuda toolkit
version_ge() {  # version_ge A B -> true if A >= B (dotted numeric)
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}
major_minor() { echo "$1" | awk -F. '{print $1"."($2==""?0:$2)}'; }

nvrtc_lib_dir() {  # prints the directory holding libnvrtc.so under a CUDA home, or nothing
    local home=$1 dir
    for dir in "$home/targets/x86_64-linux/lib" "$home/lib64"; do
        if [ -e "$dir/libnvrtc.so" ]; then echo "$dir"; return; fi
    done
}

cuda_home_version() {  # prints the toolkit version of a CUDA home (nvcc, version.json, or dir name)
    local home=$1 v=""
    if [ -x "$home/bin/nvcc" ]; then
        v=$("$home/bin/nvcc" --version 2>/dev/null | grep -o 'release [0-9][0-9.]*' | awk '{print $2}')
    fi
    if [ -z "$v" ] && [ -f "$home/version.json" ]; then
        v=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["cuda"]["version"])' "$home/version.json" 2>/dev/null || true)
    fi
    if [ -z "$v" ]; then
        v=$(basename "$home" | grep -o '[0-9][0-9.]*' || true)
    fi
    echo "$v"
}

CUDA_SCAN_REPORT=()  # one line per toolkit found: "<home> <version> <verdict>"
find_cuda_home() {  # prints the highest-versioned CUDA home with libnvrtc and 12.8 <= version <= driver CUDA, or nothing
    local candidates=() seen=() dir nvcc best="" best_v="" v mm driver_mm
    driver_mm=$(major_minor "$DRIVER_CUDA")
    CUDA_SCAN_REPORT=()
    for dir in /usr/local/cuda-1[23].* /usr/local/cuda /usr/lib/cuda; do
        [ -d "$dir" ] && candidates+=("$(readlink -f "$dir")")
    done
    if nvcc=$(command -v nvcc 2>/dev/null); then
        candidates+=("$(readlink -f "$(dirname "$nvcc")/..")")
    fi
    for dir in "${candidates[@]}"; do
        case " ${seen[*]:-} " in *" $dir "*) continue ;; esac
        seen+=("$dir")
        v=$(cuda_home_version "$dir")
        if [ -z "$(nvrtc_lib_dir "$dir")" ]; then
            CUDA_SCAN_REPORT+=("$dir ${v:-?} REJECT: no libnvrtc.so"); continue
        fi
        if [ -z "$v" ]; then
            CUDA_SCAN_REPORT+=("$dir ? REJECT: version unknown"); continue
        fi
        mm=$(major_minor "$v")
        if ! version_ge "$mm" "$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR"; then
            CUDA_SCAN_REPORT+=("$dir $v REJECT: < $MIN_CUDA_MAJOR.$MIN_CUDA_MINOR (no sm_100)"); continue
        fi
        if ! version_ge "$driver_mm" "$mm"; then
            CUDA_SCAN_REPORT+=("$dir $v REJECT: newer than driver CUDA $DRIVER_CUDA (PTX would be unsupported)"); continue
        fi
        CUDA_SCAN_REPORT+=("$dir $v OK")
        if [ -z "$best" ] || { version_ge "$v" "$best_v" && [ "$v" != "$best_v" ]; }; then
            best=$dir
            best_v=$v
        fi
    done
    [ -n "$best" ] && echo "$best"
}

print_cuda_scan() {
    local line
    log "cuda-toolkit: driver CUDA $DRIVER_CUDA; acceptable toolkit window [$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR, $(major_minor "$DRIVER_CUDA")]"
    [ "${#CUDA_SCAN_REPORT[@]}" -gt 0 ] || log "cuda-toolkit:   (no CUDA homes found under /usr/local/cuda*, /usr/lib/cuda, nvcc)"
    for line in "${CUDA_SCAN_REPORT[@]}"; do log "cuda-toolkit:   $line"; done
}

apt_install() {  # apt_install pkg... (NVIDIA repo keyring added on demand)
    [ "$HAVE_SUDO" -eq 1 ] || fail "need sudo to apt-get install $*"
    if [ "$(dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null || true)" != "install ok installed" ]; then
        local deb
        deb=$(mktemp /tmp/cuda-keyring.XXXXXX.deb)
        curl -fsSL "$CUDA_KEYRING_URL" -o "$deb"
        as_root env DEBIAN_FRONTEND=noninteractive dpkg -i "$deb"
        rm -f "$deb"
    fi
    as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -qq update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq install "$@"
}

CUDA_HOME_RESOLVED=""
CUDA_VERSION_RESOLVED=""
NVRTC_DIR=""
ENV_FILE=""
HEADER_STATUS=""
LIBCUDA_STATUS=""
LIBNVRTC_STATUS=""
COMPAT_DIR=$HOME/.river-cuda-compat   # no-sudo home for unversioned libcuda.so / libnvrtc.so symlinks
COMPAT_NEEDED=0
step_cuda() {
    local home
    if [ -n "${CUDA_HOME_OVERRIDE:-}" ]; then
        home=$(readlink -f "$CUDA_HOME_OVERRIDE")
        [ -d "$home" ] || fail "CUDA_HOME_OVERRIDE=$CUDA_HOME_OVERRIDE is not a directory"
        [ -n "$(nvrtc_lib_dir "$home")" ] || [ -e "$home/lib64/libnvrtc.so.12" ] || [ -e "$home/targets/x86_64-linux/lib/libnvrtc.so.12" ] \
            || fail "CUDA_HOME_OVERRIDE=$home has no libnvrtc.so(.12) under lib64 or targets/x86_64-linux/lib"
        skip_step cuda-toolkit "CUDA_HOME_OVERRIDE=$home ($(cuda_home_version "$home")); driver CUDA $DRIVER_CUDA"
    elif home=$(find_cuda_home) && [ -n "$home" ]; then
        print_cuda_scan
        skip_step cuda-toolkit "found $home ($(cuda_home_version "$home")) with $(nvrtc_lib_dir "$home")/libnvrtc.so; driver CUDA $DRIVER_CUDA"
    else
        print_cuda_scan
        [ "$HAVE_SUDO" -eq 1 ] || fail "no CUDA toolkit in [$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR, driver CUDA $DRIVER_CUDA] with libnvrtc found and no sudo to install $CUDA_TOOLKIT_PKG"
        log "cuda-toolkit: no toolkit in [$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR, $DRIVER_CUDA] with libnvrtc; installing $CUDA_TOOLKIT_PKG from the NVIDIA apt repo"
        apt_install "$CUDA_TOOLKIT_PKG"
        home=/usr/local/cuda-$MIN_CUDA_MAJOR.$MIN_CUDA_MINOR
        [ -d "$home" ] || fail "installed $CUDA_TOOLKIT_PKG but $home does not exist afterwards"
        [ -n "$(nvrtc_lib_dir "$home")" ] || fail "installed $CUDA_TOOLKIT_PKG but $home has no libnvrtc.so"
        done_step cuda-toolkit "installed $CUDA_TOOLKIT_PKG -> $home"
    fi
    CUDA_HOME_RESOLVED=$home
    CUDA_VERSION_RESOLVED=$(cuda_home_version "$home")
    log "cuda-toolkit: CUDA_HOME=$CUDA_HOME_RESOLVED ($CUDA_VERSION_RESOLVED)"
    check_nvrtc_headers
    check_libnvrtc_symlink
    check_libcuda_symlink
    write_cuda_env
}

missing_nvrtc_headers() {  # prints the missing entries of NVRTC_HEADERS under $CUDA_HOME_RESOLVED/include
    local h
    for h in "${NVRTC_HEADERS[@]}"; do
        case "$h" in
            */) [ -d "$CUDA_HOME_RESOLVED/include/$h" ] || echo "$h" ;;
            *)  [ -r "$CUDA_HOME_RESOLVED/include/$h" ] || echo "$h" ;;
        esac
    done
}

check_nvrtc_headers() {  # NVRTC reads these at run time from $CUDA_PATH/include (cubecl-cpp dialect.rs:62-68)
    local missing pkg_v
    missing=$(missing_nvrtc_headers)
    if [ -z "$missing" ]; then
        HEADER_STATUS="all present (${NVRTC_HEADERS[*]})"
        skip_step cuda-headers "$CUDA_HOME_RESOLVED/include: $HEADER_STATUS"
        return
    fi
    pkg_v=$(major_minor "$CUDA_VERSION_RESOLVED" | tr . -)
    local pkgs=("cuda-nvrtc-dev-$pkg_v" "cuda-cudart-dev-$pkg_v" "cuda-crt-$pkg_v" "cuda-cccl-$pkg_v")
    log "cuda-headers: missing in $CUDA_HOME_RESOLVED/include: $(echo "$missing" | tr '\n' ' ')-> installing ${pkgs[*]}"
    [ "$HAVE_SUDO" -eq 1 ] || fail "cuda-headers: missing $(echo "$missing" | tr '\n' ' ') and no sudo to install ${pkgs[*]}"
    apt_install "${pkgs[@]}"
    missing=$(missing_nvrtc_headers)
    [ -z "$missing" ] || fail "cuda-headers: still missing after installing ${pkgs[*]}: $(echo "$missing" | tr '\n' ' ') (NVRTC will fail to compile every kernel)"
    HEADER_STATUS="all present after installing ${pkgs[*]}"
    done_step cuda-headers "$HEADER_STATUS"
}

ensure_unversioned_so() {  # ensure_unversioned_so <label> <dir> <versioned-file> -> sets REPLY to a status string
    # cudarc 0.12.1 dlopens the unversioned name only (lib.rs:101-128, driver/sys/mod.rs:64-67).
    local label=$1 dir=$2 versioned=$3 target
    if [ -e "$dir/$label" ]; then
        REPLY="$dir/$label present ($(readlink -f "$dir/$label"))"
        return
    fi
    target=$(readlink -f "$dir/$versioned")
    if [ "$HAVE_SUDO" -eq 1 ]; then
        as_root ln -s "$versioned" "$dir/$label"
        as_root ldconfig 2>/dev/null || true
        REPLY="created $dir/$label -> $versioned (sudo)"
    else
        mkdir -p "$COMPAT_DIR"
        ln -sfn "$target" "$COMPAT_DIR/$label"
        COMPAT_NEEDED=1
        REPLY="no sudo: created $COMPAT_DIR/$label -> $target (prepended to LD_LIBRARY_PATH)"
    fi
}

check_libnvrtc_symlink() {
    local dir
    NVRTC_DIR=$(nvrtc_lib_dir "$CUDA_HOME_RESOLVED")
    if [ -n "$NVRTC_DIR" ]; then
        LIBNVRTC_STATUS="$NVRTC_DIR/libnvrtc.so present ($(readlink -f "$NVRTC_DIR/libnvrtc.so"))"
        skip_step libnvrtc "$LIBNVRTC_STATUS"
        return
    fi
    for dir in "$CUDA_HOME_RESOLVED/targets/x86_64-linux/lib" "$CUDA_HOME_RESOLVED/lib64"; do
        [ -e "$dir/libnvrtc.so.12" ] || continue
        ensure_unversioned_so libnvrtc.so "$dir" libnvrtc.so.12
        LIBNVRTC_STATUS=$REPLY
        NVRTC_DIR=$(nvrtc_lib_dir "$CUDA_HOME_RESOLVED")
        [ -n "$NVRTC_DIR" ] || NVRTC_DIR=$dir
        done_step libnvrtc "$LIBNVRTC_STATUS"
        return
    done
    fail "libnvrtc: neither libnvrtc.so nor libnvrtc.so.12 under $CUDA_HOME_RESOLVED/{lib64,targets/x86_64-linux/lib}"
}

check_libcuda_symlink() {
    local ldc dir versioned
    ldc=$(ldconfig -p 2>/dev/null | grep -E 'libcuda\.so$' | head -n1 | awk '{print $NF}' || true)
    if [ -n "$ldc" ] && [ -e "$ldc" ]; then
        LIBCUDA_STATUS="$ldc in ldconfig cache ($(readlink -f "$ldc"))"
        skip_step libcuda "$LIBCUDA_STATUS"
        return
    fi
    if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
        LIBCUDA_STATUS="/usr/lib/x86_64-linux-gnu/libcuda.so present ($(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so)), not in ldconfig cache"
        skip_step libcuda "$LIBCUDA_STATUS"
        return
    fi
    # Only the versioned driver name exists: find its directory (never the toolkit's lib/stubs).
    versioned=$(ldconfig -p 2>/dev/null | grep -E 'libcuda\.so\.1 ' | head -n1 | awk '{print $NF}' || true)
    if [ -z "$versioned" ]; then
        for dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu; do
            [ -e "$dir/libcuda.so.1" ] && { versioned=$dir/libcuda.so.1; break; }
        done
    fi
    [ -n "$versioned" ] || fail "libcuda: no libcuda.so or libcuda.so.1 on this box (driver userspace missing; not installing drivers)"
    dir=$(dirname "$versioned")
    log "libcuda: only $versioned exists; cudarc needs the unversioned libcuda.so"
    ensure_unversioned_so libcuda.so "$dir" libcuda.so.1
    LIBCUDA_STATUS=$REPLY
    done_step libcuda "$LIBCUDA_STATUS"
}

write_cuda_env() {
    local content compat_line=""
    if [ "$COMPAT_NEEDED" -eq 1 ] || [ -e "$COMPAT_DIR/libcuda.so" ] || [ -e "$COMPAT_DIR/libnvrtc.so" ]; then
        compat_line="case \":\${LD_LIBRARY_PATH:-}:\" in *\":$COMPAT_DIR:\"*) ;; *) export LD_LIBRARY_PATH=$COMPAT_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} ;; esac"
    fi
    content=$(cat <<EOF
# River Song CUDA environment (written by b200_bootstrap.sh; re-run it to regenerate)
export CUDA_PATH=$CUDA_HOME_RESOLVED
export CUDA_HOME=$CUDA_HOME_RESOLVED
case ":\$PATH:" in *":$CUDA_HOME_RESOLVED/bin:"*) ;; *) export PATH=$CUDA_HOME_RESOLVED/bin:\$PATH ;; esac
case ":\${LD_LIBRARY_PATH:-}:" in *":$NVRTC_DIR:"*) ;; *) export LD_LIBRARY_PATH=$NVRTC_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} ;; esac
${compat_line}
export JEV_PCN_CUDA_LIB_DIR=$NVRTC_DIR
unset JEV_PCN_CUDA_RUNTIME_READY
EOF
    )
    # ~/.river-b200.env always (non-login ssh commands do not read /etc/profile.d; `source ~/.river-b200.env` there).
    local user_env=$HOME/.river-b200.env wrote_user=0
    if [ ! -f "$user_env" ] || [ "$(cat "$user_env")" != "$content" ]; then
        printf '%s\n' "$content" > "$user_env"
        wrote_user=1
    fi
    if [ "$HAVE_SUDO" -eq 1 ]; then
        ENV_FILE=/etc/profile.d/river-cuda.sh
        if [ -f "$ENV_FILE" ] && [ "$(cat "$ENV_FILE")" = "$content" ]; then
            skip_step cuda-env "$ENV_FILE up to date"
        else
            printf '%s\n' "$content" | as_root tee "$ENV_FILE" >/dev/null
            as_root chmod 0644 "$ENV_FILE"
            done_step cuda-env "wrote $ENV_FILE (login shells) and $user_env"
        fi
    else
        ENV_FILE=$user_env
        if [ "$wrote_user" -eq 1 ]; then
            done_step cuda-env "no sudo: wrote $ENV_FILE; source it in every shell"
        else
            skip_step cuda-env "no sudo: $ENV_FILE up to date"
        fi
    fi
    # shellcheck disable=SC1090
    source "$ENV_FILE"
}

# ---------------------------------------------------------------- 5. rustup
step_rustup() {
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        skip_step rustup "$HOME/.cargo/bin/cargo present"
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
        done_step rustup "installed stable (minimal)"
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
}

# ---------------------------------------------------------------- 6. clone
step_clone() {
    if [ ! -d "$USER_HOME/Dev" ]; then
        if [ -w "$(dirname "$USER_HOME")" ]; then
            mkdir -p "$USER_HOME/Dev"
        else
            as_root mkdir -p "$USER_HOME/Dev"
            as_root chown "$LOGIN_USER:" "$USER_HOME" "$USER_HOME/Dev"
        fi
        log "clone: created $USER_HOME/Dev owned by $LOGIN_USER"
    elif [ ! -w "$USER_HOME/Dev" ]; then
        as_root chown "$LOGIN_USER:" "$USER_HOME" "$USER_HOME/Dev"
    fi
    if [ -d "$REPO_DIR/.git" ]; then
        git -C "$REPO_DIR" fetch -q origin
        if [ -n "${REPO_REF:-}" ]; then
            git -C "$REPO_DIR" checkout -q --detach "$REPO_REF" 2>/dev/null || git -C "$REPO_DIR" checkout -q --detach "origin/$REPO_REF"
            done_step clone "existing checkout updated to $REPO_REF"
        else
            git -C "$REPO_DIR" checkout -q main
            git -C "$REPO_DIR" pull -q --ff-only origin main
            done_step clone "existing checkout fast-forwarded to origin/main"
        fi
    else
        git clone -q "$REPO_URL" "$REPO_DIR"
        if [ -n "${REPO_REF:-}" ]; then
            git -C "$REPO_DIR" checkout -q --detach "$REPO_REF" 2>/dev/null || git -C "$REPO_DIR" checkout -q --detach "origin/$REPO_REF"
        fi
        done_step clone "cloned $REPO_URL -> $REPO_DIR @ ${REPO_REF:-main}"
    fi
    log "clone: commit $(git -C "$REPO_DIR" rev-parse --short HEAD) ($(git -C "$REPO_DIR" log -1 --format=%cd --date=iso))"
}

# ---------------------------------------------------------------- 7. build
step_build() {
    if [ "${SKIP_BUILD:-0}" = "1" ]; then
        skip_step build "SKIP_BUILD=1"
        return
    fi
    local jobs targets=() name
    jobs=$(nproc)
    if [ "${BUILD_ALL:-0}" = "1" ]; then
        targets=(--bins)
        BINARIES=("${BUILD_ALL_BINARIES[@]}")
    else
        for name in "${BINARIES[@]}"; do targets+=(--bin "$name"); done
    fi
    log "build: cargo build --locked --release --features cuda ${targets[*]} --example gpu_smoke with $jobs jobs in $REPO_DIR"
    ( cd "$REPO_DIR" && CARGO_BUILD_JOBS=$jobs cargo build --locked --release --features cuda "${targets[@]}" --example gpu_smoke )
    mkdir -p "$LIVE_DIR"
    local copied=0 present=0
    for name in "${BINARIES[@]}"; do
        local src=$REPO_DIR/target/release/$name
        [ -x "$src" ] || { log "build: $name not built (not in this checkout?)"; continue; }
        present=$((present + 1))
        if [ -x "$LIVE_DIR/$name" ] && cmp -s "$src" "$LIVE_DIR/$name"; then
            continue
        fi
        install -m 0755 "$src" "$LIVE_DIR/$name.tmp" && mv -f "$LIVE_DIR/$name.tmp" "$LIVE_DIR/$name"
        copied=$((copied + 1))
    done
    [ -x "$LIVE_DIR/river-pcn-train-universal" ] || fail "build: $LIVE_DIR/river-pcn-train-universal missing after build"
    done_step build "$present binaries built, $copied copied to $LIVE_DIR"
}

# ---------------------------------------------------------------- 2. data paths (runs right after apt: before any transfer unpack)
DATA_DISK_CHOICE=""
pick_data_disk() {  # prints the directory mount point with the most free space (excluding pseudo filesystems)
    local free mount
    df -P -x tmpfs -x devtmpfs -x squashfs -x efivarfs 2>/dev/null \
        | awk 'NR>1 && $6 !~ "^/(boot|snap|run|dev|sys|proc|etc)(/|$)" {print $4, $6}' \
        | sort -n -r \
        | while read -r free mount; do
            # containers bind-mount single files (e.g. /etc/hosts) from the host disk: not a disk
            [ -d "$mount" ] || continue
            echo "$mount"; break
        done
}

mount_of() { df -P "$1" 2>/dev/null | awk 'NR==2{print $6}'; }
free_gb_of() { df -P -BG "$1" 2>/dev/null | awk 'NR==2{sub("G","",$4); print $4}'; }

MISPLACED_DIRS=()
check_misplaced_top_level() {  # a real directory on a filesystem other than the data disk (transfer bundle ran first)
    local top=$1 here dest
    [ -d "$top" ] && [ ! -L "$top" ] || return 0
    [ "$DATA_DISK_CHOICE" != "/" ] || return 0
    here=$(mount_of "$top")
    [ "$here" != "$DATA_DISK_CHOICE" ] || return 0
    dest=$DATA_DISK_CHOICE$top
    echo "[bootstrap] WARNING: $top is a REAL directory on filesystem '$here', not on the data disk $DATA_DISK_CHOICE" >&2
    echo "[bootstrap] WARNING:   size: $(du -sh "$top" 2>/dev/null | awk '{print $1}')   free on $here: $(df -h "$here" | awk 'NR==2{print $4}')   free on $DATA_DISK_CHOICE: $(df -h "$DATA_DISK_CHOICE" | awk 'NR==2{print $4}')" >&2
    echo "[bootstrap] WARNING:   to move it (or re-run with MOVE_DATA_DIRS=1):" >&2
    echo "[bootstrap] WARNING:     sudo mv $top $dest && sudo ln -s $dest $top" >&2
    if [ "${MOVE_DATA_DIRS:-0}" = "1" ]; then
        [ ! -e "$dest" ] || fail "data: MOVE_DATA_DIRS=1 but $dest already exists; merge by hand"
        as_root mkdir -p "$(dirname "$dest")"
        as_root mv "$top" "$dest"
        as_root ln -s "$dest" "$top"
        log "data: moved $top -> $dest (MOVE_DATA_DIRS=1)"
    else
        MISPLACED_DIRS+=("$top on $here")
    fi
}

ensure_top_level() {  # ensure_top_level /bulk-storage
    local top=$1
    if [ -e "$top" ] || [ -L "$top" ]; then
        log "data: $top exists ($(readlink -f "$top"))"
        return
    fi
    if [ "$DATA_DISK_CHOICE" = "/" ]; then
        as_root mkdir -p "$top"
    else
        as_root mkdir -p "$DATA_DISK_CHOICE${top}"
        as_root chown "$LOGIN_USER:" "$DATA_DISK_CHOICE${top}"
        as_root ln -s "$DATA_DISK_CHOICE${top}" "$top"
        log "data: $top -> $DATA_DISK_CHOICE${top}"
    fi
    as_root chown "$LOGIN_USER:" "$top"
}

DATA_FREE_GB=""
step_data_paths() {
    if [ -n "${DATA_DISK:-}" ]; then
        [ -d "$DATA_DISK" ] || fail "DATA_DISK=$DATA_DISK is not a directory"
        DATA_DISK_CHOICE=$(readlink -f "$DATA_DISK")
        log "data: DATA_DISK=$DATA_DISK_CHOICE (from env)"
    else
        DATA_DISK_CHOICE=$(pick_data_disk)
        [ -n "$DATA_DISK_CHOICE" ] || DATA_DISK_CHOICE=/
        log "data: auto-detected largest free filesystem: $DATA_DISK_CHOICE"
    fi
    df -h
    check_misplaced_top_level /bulk-storage
    check_misplaced_top_level /fast-storage
    ensure_top_level /bulk-storage
    ensure_top_level /fast-storage
    local dir created=0
    for dir in "${DATA_DIRS[@]}"; do
        [ -d "$dir" ] && continue
        mkdir -p "$dir" 2>/dev/null || as_root mkdir -p "$dir"
        [ "$(stat -c %U "$dir")" = "$LOGIN_USER" ] || as_root chown "$LOGIN_USER:" "$dir"
        created=$((created + 1))
    done
    for dir in /bulk-storage /fast-storage /bulk-storage/connectome-merc /bulk-storage/datasets /fast-storage/river-datasets; do
        [ "$(stat -L -c %U "$dir")" = "$LOGIN_USER" ] || as_root chown "$LOGIN_USER:" "$dir"
    done
    DATA_FREE_GB=$(free_gb_of "$DATA_DISK_CHOICE")
    log "data: data disk $DATA_DISK_CHOICE has ${DATA_FREE_GB:-?} GB free"
    if [ "${DATA_FREE_GB:-0}" -lt "$MIN_DATA_FREE_GB" ]; then
        echo "[bootstrap] WARNING: only ${DATA_FREE_GB:-?} GB free on $DATA_DISK_CHOICE (< $MIN_DATA_FREE_GB GB); a 16384-wide run root + ladder scratch needs tens of GB, 49152-wide ~250 GB" >&2
    fi
    df -h /bulk-storage/ /fast-storage/ "$USER_HOME"
    done_step data-paths "$created of ${#DATA_DIRS[@]} dirs created, owner $LOGIN_USER, disk $DATA_DISK_CHOICE ${DATA_FREE_GB:-?} GB free${MISPLACED_DIRS:+; MISPLACED: ${MISPLACED_DIRS[*]}}"
}

# ---------------------------------------------------------------- 8. smoke
SMOKE_RESULT=""
WMMA_STATUS=""
CUBECL_SMOKE_LOG=/tmp/cubecl-smoke.log
step_smoke() {
    if [ "${SKIP_SMOKE:-0}" = "1" ]; then
        skip_step smoke "SKIP_SMOKE=1"
        return
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    local exe=$LIVE_DIR/river-pcn-train-universal
    [ -x "$exe" ] || fail "smoke: $exe missing (run without SKIP_BUILD=1 first)"
    local help_text
    help_text=$("$exe" --help) || fail "smoke: $exe --help failed"
    printf '%s\n' "$help_text" | head -n 3
    log "smoke: trainer --help ok ($(printf '%s\n' "$help_text" | wc -l) lines)"
    # CUBECL_DEBUG_LOG=<path> makes cubecl-runtime 0.4.0 (src/debug.rs:278,334) write kernel sources/compile logs there.
    rm -f "$CUBECL_SMOKE_LOG"
    local smoke_bin=$REPO_DIR/target/release/examples/gpu_smoke
    if [ -x "$smoke_bin" ]; then
        SMOKE_RESULT=$(CUBECL_DEBUG_LOG=$CUBECL_SMOKE_LOG timeout 600 "$smoke_bin") || fail "smoke: gpu_smoke failed (cubecl JIT on this GPU); see $CUBECL_SMOKE_LOG"
    else
        SMOKE_RESULT=$(cd "$REPO_DIR" && CUBECL_DEBUG_LOG=$CUBECL_SMOKE_LOG timeout 1800 cargo run -q --locked --release --features cuda --example gpu_smoke) \
            || fail "smoke: gpu_smoke failed (cubecl JIT on this GPU); see $CUBECL_SMOKE_LOG"
    fi
    echo "$SMOKE_RESULT"
    if [ -s "$CUBECL_SMOKE_LOG" ]; then
        if grep -q wmma "$CUBECL_SMOKE_LOG"; then
            WMMA_STATUS="wmma (tensor-core) kernel source present in $CUBECL_SMOKE_LOG ($(grep -c wmma "$CUBECL_SMOKE_LOG") lines)"
        else
            WMMA_STATUS="no 'wmma' in $CUBECL_SMOKE_LOG: autotune did not compile a tensor-core matmul for this shape (report only)"
        fi
    else
        WMMA_STATUS="$CUBECL_SMOKE_LOG empty/missing: cubecl wrote no debug log (report only)"
    fi
    log "smoke: $WMMA_STATUS"
    done_step smoke "$SMOKE_RESULT"
}

# ---------------------------------------------------------------- 9. summary
step_summary() {
    echo
    echo "================ b200_bootstrap summary ================"
    local line
    for line in "${SUMMARY[@]}"; do echo "  $line"; done
    echo "  user: $LOGIN_USER (sudo: $HAVE_SUDO)  host: $(hostname)  cores: $(nproc)"
    echo "  driver: $DRIVER_VERSION  driver CUDA: $DRIVER_CUDA  $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | head -n1)"
    echo "  nvcc: $( ("${CUDA_HOME_RESOLVED:-/nonexistent}/bin/nvcc" --version 2>/dev/null || echo 'nvcc: not found') | grep -o 'release.*' || true)"
    echo "  CUDA_HOME: ${CUDA_HOME_RESOLVED:-?} ($CUDA_VERSION_RESOLVED)  env: ${ENV_FILE:-?}"
    echo "  libnvrtc: ${LIBNVRTC_STATUS:-?}"
    echo "  headers ($CUDA_HOME_RESOLVED/include): ${HEADER_STATUS:-?}"
    echo "  libcuda.so: ${LIBCUDA_STATUS:-?}"
    echo "  data disk: $DATA_DISK_CHOICE (${DATA_FREE_GB:-?} GB free)  /bulk-storage -> $(readlink -f /bulk-storage)  /fast-storage -> $(readlink -f /fast-storage)"
    [ "${#MISPLACED_DIRS[@]}" -eq 0 ] || echo "  WARNING misplaced data dirs (see above for move commands): ${MISPLACED_DIRS[*]}"
    echo "  rustc: $(rustc --version 2>/dev/null || echo '?')  cargo: $(cargo --version 2>/dev/null || echo '?')"
    echo "  repo: $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
    if [ -d "$LIVE_DIR" ]; then
        echo "  binaries ($LIVE_DIR):"
        local file
        for file in "$LIVE_DIR"/*; do if [ -f "$file" ]; then sha256sum "$file" | sed 's/^/    /'; fi; done
    fi
    [ -z "$SMOKE_RESULT" ] || echo "  smoke: $SMOKE_RESULT"
    [ -z "$WMMA_STATUS" ] || echo "  wmma: $WMMA_STATUS"
    echo "========================================================="
}

step_apt
step_data_paths
step_driver
step_cuda
step_rustup
step_clone
step_build
step_smoke
step_summary
