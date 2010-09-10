#!/bin/bash


gnuplot plot.gp
montage 460GTX*.png -geometry 800x600\>+2+4 samlet_460.png
montage 8800*.png -geometry 800x600\>+2+4 samlet_8800.png