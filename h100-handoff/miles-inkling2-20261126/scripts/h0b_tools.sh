#!/bin/bash
# H0b (round 3), on a node with internet, outside the container: py-spy 0.4.0 as a standalone binary in <WORK>/tools,
# used only by hang_stacks.sh to read the Python stacks of a hung trainer. Nothing is installed into the image.
# usage: WORK=<WORK> bash h0b_tools.sh
set -euo pipefail
: "${WORK:?}"
mkdir -p "$WORK/tools" "$WORK/runs/h0"
cd "$WORK/tools"
WHL=py_spy-0.4.0-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl
python3 -m pip download py-spy==0.4.0 --no-deps --only-binary=:all: --platform manylinux1_x86_64 -d . -q
echo "8bf2f3702cef367a489faa45177b41a6c31b2a3e5bd78c978d44e29340152f5a  $WHL" | sha256sum -c - | tee "$WORK/runs/h0/py-spy.sha256"
python3 -c "import sys, zipfile; open('py-spy', 'wb').write(zipfile.ZipFile(sys.argv[1]).read('py_spy-0.4.0.data/scripts/py-spy'))" "$WHL"
chmod +x py-spy
./py-spy --version | tee -a "$WORK/runs/h0/py-spy.sha256"
