#!/bin/sh
convert -delay 5 -loop 0 anim/frame*.png anim/animation.gif
rm -I anim/frame*.png
