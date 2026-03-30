#!/usr/bin/env sh
BASEDIR=$(dirname "$0")
PORT=${1:-8000}
NAME=${2:-mango}
BUILDER=$(command -v podman || command -v docker)
$BUILDER build --build-arg PORT=$PORT -t $NAME -f $BASEDIR/Dockerfile $BASEDIR/../
