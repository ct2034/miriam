#!/bin/bash
declare -a xcf_files=("intro" "scenario" "random")
declare -a png_files=("intro" "scenario" "random" "scenario_problem" "random_problem")
declare -a slow_files=("scenario_minlp" "scenario_tcbs" "random_minlp" "random_tcbs")
declare duration_text=5
declare outfile="iros2018.mp4"

echo "pwd: $PWD"

for file in "${xcf_files[@]}"; do
	xcf2png $file.xcf -o $file.png
done
for file in "${png_files[@]}"; do
  ffmpeg -y -loop 1 -i $file.png -c:v libx264 -t $duration_text -r 10 -pix_fmt yuv420p $file.mp4
done
for file in "${slow_files[@]}"; do
  ffmpeg -y -i $file.mp4 -filter:v "setpts=2.0*PTS" ${file}_sl.mp4
done

ffmpeg -y -f concat -safe 0 -i track_list.txt -c copy $outfile

cvlc --no-loop $outfile
