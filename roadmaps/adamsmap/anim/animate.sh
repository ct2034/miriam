#!/bin/sh
convert -delay 5 -loop 0 frame*.png animation.gif
rm -I frame*.png
