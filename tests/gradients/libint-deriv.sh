#!/usr/bin/env bash
#
# Generate and compile a libint with FIRST derivatives, outside the image.
#
# The production libint (dockerbuild/Dockerfile) is energy-only: derivative order 0, MAX_AM=7.
# Gradients need order 1 of the 1-body integrals and of the 4-, 3- and 2-center ERI, at the same
# MAX_AM=7 (same pinned libint commit). Generation and compilation are heavy, so this runs
# stage by stage inside the builder image with a memory cap, in a host directory (default
# ~/nopt-libint-deriv), and nothing is baked into an image.
#
#   libint-deriv.sh gen        generate the source           -> $OUT/libint-src
#   libint-deriv.sh compile    compile + install (shared)    -> $OUT/install
# Env: ARCH (x86-64-v4)  DMRG (true)  OUT  MEM (16g)  GEN_JOBS (8)  CC_JOBS (auto from RAM)
#      AM (7)  DERIV (1)  ERI4 (=DERIV)
# ERI4=0 keeps the 4-center ERI at derivative order 0 (what the energies need) and gives
# first derivatives only for the 1-body, 3-center and 2-center integrals, which is all the RI
# gradient route needs; the 4-center derivatives are by far the heaviest part of the generation.
set -euo pipefail

ARCH=${ARCH:-x86-64-v4}; DMRG=${DMRG:-true}
OUT=${OUT:-$HOME/nopt-libint-deriv}
MEM=${MEM:-16g}; GEN_JOBS=${GEN_JOBS:-8}
AM=${AM:-7}; DERIV=${DERIV:-1}; ERI4=${ERI4:-$DERIV}
IMG=nopt-grad-builder:${ARCH}-dmrg${DMRG}
REF=5fb07b4862f219749c51d59ee08073d20a56d506   # libint v2.13.1, as in dockerbuild/Dockerfile
mkdir -p "$OUT"

# "7;7" for order 1: one MAX_AM per derivative level
levels() { local i out=; for i in $(seq 0 "$1"); do out+="$AM;"; done; echo "${out%;}"; }
LEVELS=$(levels "$DERIV"); LEVELS4=$(levels "$ERI4")

run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --memory="$MEM" --memory-swap="$MEM" \
            -v "$OUT:/out" -w /out "$IMG" sh -c "$1"; }

case "${1:-}" in
gen)
    run "set -eux
        [ -d libint ] || { git clone -q https://github.com/evaleev/libint.git; git -C libint checkout -q $REF; }
        cd libint
        cmake -S . -B build -G Ninja -D CMAKE_CXX_COMPILER=\$CXX -D CMAKE_BUILD_TYPE=Release \
            -D LIBINT2_ENABLE_ONEBODY=$DERIV -D LIBINT2_ENABLE_ERI=$ERI4 \
            -D LIBINT2_ENABLE_ERI3=$DERIV -D LIBINT2_ENABLE_ERI2=$DERIV \
            -D LIBINT2_MAX_AM='$LEVELS' -D LIBINT2_ERI_MAX_AM='$LEVELS4' -D LIBINT2_ERI3_MAX_AM='$LEVELS' -D LIBINT2_ERI2_MAX_AM='$LEVELS' \
            -D LIBINT2_ENABLE_GENERIC_CODE=ON
        cmake --build build --target export -j $GEN_JOBS
        d=\$(find build -maxdepth 1 -type d -name 'libint-*' | head -1); test -n \"\$d\"
        rm -rf /out/libint-src; cp -a \"\$d\" /out/libint-src" ;;
compile)
    run "set -eux; MARCH=\$(cat /opt/arch/march_flags)
        cmake -S /out/libint-src -B /out/libint-src/build -G Ninja -D CMAKE_CXX_COMPILER=\$CXX \
            -D CMAKE_INSTALL_PREFIX=/out/install -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS=\"\$MARCH\" \
            -D BUILD_SHARED_LIBS=ON -D BUILD_TESTING=OFF \
            -D LIBINT2_REQUIRE_CXX_API=ON -D LIBINT2_REQUIRE_CXX_API_COMPILED=ON
        cmake --build /out/libint-src/build --target install -j \"\${CC_JOBS:-\$(build-jobs 1024)}\"" ;;
*)  sed -n '2,15p' "$0"; exit 2 ;;
esac
