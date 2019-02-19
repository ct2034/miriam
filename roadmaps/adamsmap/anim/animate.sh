#!/bin/sh
convert -delay 10 -loop 0 frame*.png animation.gif
rm -I frame*.png
