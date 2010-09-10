#!/bin/bash


gnuplot plot.gp
montage 460GTX_*.png -geometry 800x600\>+2+4 samlet_460.png
montage 8800_*.png -geometry 800x600\>+2+4 samlet_8800.png