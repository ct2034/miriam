#!/bin/sh
while inotifywait -e modify *; do
  ./test.sh
done