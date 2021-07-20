#! /bin/bash

# PURPOSE:
#
# run pdflatex inside a docker container to avoid
# caring about tex package management
#
# USAGE:
#
# Ensure the docker daemon is running.
#
# Ensure your working directory contains this script
# and directories named src/ and out/ (these will be
# bind-mounted into the container for build input and
# build output respectively).
#
# ./gutenbot_pdflatex.sh src/somefile.tex

set -euo pipefail

docker build -t gutenbot:dev -f gutenbot.Dockerfile .

docker run --rm \
  --name gutenbot \
  --mount type=bind,source="$(pwd)"/src,target=/work/src \
  --mount type=bind,source="$(pwd)"/out,target=/work/out \
  --entrypoint /bin/pdflatex \
  gutenbot:dev \
  -interaction nonstopmode \
  -output-directory /work/out \
  -output-format pdf \
  /work/$1


