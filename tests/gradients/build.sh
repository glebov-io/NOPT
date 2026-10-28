#!/usr/bin/env bash
#
# Build NOPT from this working tree inside the NOPT builder image (Docker only).
#
# The dependencies (libint, libgrpp, block2, OpenBLAS) come from the `builder` stage of
# dockerbuild/Dockerfile; they are cached by Docker, so the image step is fast after the first
# time. NOPT itself is then compiled from the working tree, bind-mounted, so make rebuilds
# incrementally. The programs run only inside that image (see run.sh).
#
# Knobs (env): ARCH=x86-64-v4  GTO_MAX=27  DMRG=true|false  JOBS=auto
#   TARGETS="run_sQM ..." (default: the programs the tests use)
# Writes ./Makefile.vars (gitignored) if absent, from the one the image was built with.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ARCH=${ARCH:-x86-64-v4}
GTO_MAX=${GTO_MAX:-27}
DMRG=${DMRG:-true}
JOBS=${JOBS:-auto}
TARGETS=${TARGETS:-run_sQM}
# Only keys the dependency layers; the NOPT that stage compiles is never used.
BASE_REF=fc25396f446d8211407ed9ca0832bb476287bf61
IMG=nopt-grad-builder:${ARCH}-dmrg${DMRG}

docker buildx build -f "$ROOT/dockerbuild/Dockerfile" "$ROOT/dockerbuild" --target builder --load \
    -t "$IMG" --build-arg ARCH="$ARCH" --build-arg GTO_MAX="$GTO_MAX" --build-arg DMRG="$DMRG" \
    --build-arg NOPT_GIT_REF="$BASE_REF" --progress=plain > "${TMPDIR:-/tmp}/nopt-grad-builder.log" 2>&1 \
    || { tail -20 "${TMPDIR:-/tmp}/nopt-grad-builder.log" >&2; exit 1; }

[ -f "$ROOT/Makefile.vars" ] || docker run --rm "$IMG" cat /deps/NOPT/Makefile.vars > "$ROOT/Makefile.vars"

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="$JOBS" -v "$ROOT:/work" -w /work "$IMG" sh -c '
    set -e
    export NOPT_ARCH_FLAGS="$(cat /opt/arch/march_flags)"
    if [ "'"$DMRG"'" = true ]; then B2=yes; else B2=no; fi
    make CXX="$CXX" USE_BLOCK2="$B2" GTO_MAX='"$GTO_MAX"' '"$TARGETS"' -j"$(build-jobs 2560)"'
echo "built $TARGETS  (image $IMG)"
