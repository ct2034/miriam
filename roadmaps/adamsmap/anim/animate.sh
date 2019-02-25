#!/bin/sh
convert -delay 50 -loop 0 frame*.png animation.gif
rm -I frame*.png
