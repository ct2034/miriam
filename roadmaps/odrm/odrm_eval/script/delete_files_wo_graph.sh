#!/bin/sh
for f in res/*.pkl
do
  echo "\n> Processing $f file..."
  res=$(script/transform_pickle.py list $f)
  echo $res
  echo $res | grep --quiet "posar"
  if [ $? = 1 ]; then
    echo "Deleting .."
    rm $f
  else
    echo "Keeping .."
  fi
done
