#!/usr/bin/env bash
#
# Run a program built by build.sh inside its builder image.
#   [PROG=run_sQM] run.sh <dir> <input> [args...]    (runs in <dir>, as the calling user)
# Env: ARCH, DMRG as in build.sh; NOPT_ENERGY_JSON=<file> (relative to <dir>) makes the run write
# its full-precision energies there.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
IMG=nopt-grad-builder:${ARCH:-x86-64-v4}-dmrg${DMRG:-true}
LIBDIRS=/opt/libint/lib:/opt/block2/lib; MOUNT=
PROG=${PROG:-run_sQM}
DIR=$(cd "$1" && pwd); INP=$2; shift 2
# NOPT_MEASURE=1: report the wall time and the peak resident memory of the program (stderr)
WRAP=()
if [ -n "${NOPT_MEASURE:-}" ]; then
    WRAP=(python3 -c 'import resource,subprocess,sys,time; t=time.time(); r=subprocess.run(sys.argv[1:]).returncode; sys.stderr.write("NOPT_MEASURE wall %.1f s, peak RSS %.0f MB\n" % (time.time()-t, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024.0)); sys.exit(r)')
fi

# every NOPT_* variable of the caller is passed into the container (switches, test knobs, output files)
ENVS=(); while IFS='=' read -r k _; do [[ $k == NOPT_* ]] && ENVS+=(-e "$k"); done < <(env)
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp "${ENVS[@]}" \
    --shm-size=4g -e LD_LIBRARY_PATH=$LIBDIRS $MOUNT -v "$ROOT:/work:ro" -v "$DIR:/data" -w /data "$IMG" \
    "${WRAP[@]}" /work/$PROG -i "$INP" "$@"
