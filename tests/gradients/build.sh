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
#   LIBINT_DERIV=<install prefix of a libint with first derivatives, see libint-deriv.sh>
#       default ~/nopt-libint-deriv/install if it exists; set empty to use the energy-only libint of
#       the image. The derivative code (src/integral_derivatives.cpp) needs the former. Switching
#       between the two cleans the tree first (objects must not mix headers).
#   TARGETS="run_sQM ..." (default: the programs the tests use)
# Writes ./Makefile.vars (gitignored) if absent, from the one the image was built with.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ARCH=${ARCH:-x86-64-v4}
GTO_MAX=${GTO_MAX:-27}
DMRG=${DMRG:-true}
JOBS=${JOBS:-auto}
LIBINT_DERIV=${LIBINT_DERIV-$HOME/nopt-libint-deriv/install}
[ -d "$LIBINT_DERIV" ] || LIBINT_DERIV=
TARGETS=${TARGETS:-run_sQM run_derivtest run_ritest}
# Only keys the dependency layers; the NOPT that stage compiles is never used.
BASE_REF=fc25396f446d8211407ed9ca0832bb476287bf61
IMG=nopt-grad-builder:${ARCH}-dmrg${DMRG}

docker buildx build -f "$ROOT/dockerbuild/Dockerfile" "$ROOT/dockerbuild" --target builder --load \
    -t "$IMG" --build-arg ARCH="$ARCH" --build-arg GTO_MAX="$GTO_MAX" --build-arg DMRG="$DMRG" \
    --build-arg NOPT_GIT_REF="$BASE_REF" --progress=plain > "${TMPDIR:-/tmp}/nopt-grad-builder.log" 2>&1 \
    || { tail -20 "${TMPDIR:-/tmp}/nopt-grad-builder.log" >&2; exit 1; }

[ -f "$ROOT/Makefile.vars" ] || docker run --rm "$IMG" cat /deps/NOPT/Makefile.vars > "$ROOT/Makefile.vars"

MODE=${LIBINT_DERIV:-image}
MOUNT=; MAKEVARS=
if [ -n "$LIBINT_DERIV" ]; then
    MOUNT="-v $LIBINT_DERIV:/opt/libint-deriv:ro"
    MAKEVARS="LIBINT_H_DIR=/opt/libint-deriv/include LIBINT_L_DIR=/opt/libint-deriv/lib"
fi
CLEAN=
[ "$(cat "$ROOT/.libint_mode" 2>/dev/null || true)" = "$MODE" ] || CLEAN=yes

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="$JOBS" $MOUNT -v "$ROOT:/work" -w /work "$IMG" sh -c '
    set -e
    export NOPT_ARCH_FLAGS="$(cat /opt/arch/march_flags)"
    if [ "'"$DMRG"'" = true ]; then B2=yes; else B2=no; fi
    if [ -n "'"$CLEAN"'" ]; then make clean CXX="$CXX" >/dev/null; fi
    make CXX="$CXX" USE_BLOCK2="$B2" GTO_MAX='"$GTO_MAX"' '"$MAKEVARS"' '"$TARGETS"' -j"$(build-jobs 2560)"'
[ "$(cat "$ROOT/.libint_mode" 2>/dev/null || true)" = "$MODE" ] || echo "$MODE" > "$ROOT/.libint_mode"   # rewritten only on change: run.sh reads it concurrently
echo "built $TARGETS  (image $IMG, libint: $MODE)"
