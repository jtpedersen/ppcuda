set terminal png


set title "32x32 blocks"
set output "460GTX_32_blocks.png"

plot "32_460.dat" using 1:2 w linespoints title 'Simple', \
     "32_460.dat" using 1:3 w linespoints title 'Tiled', \
     "32_460.dat" using 1:4 w linespoints title 'Textured'

set title "16x16 blocks"
set output "460GTX_16_blocks.png"

plot "16_460.dat" using 1:2 w linespoints title 'Simple', \
     "16_460.dat" using 1:3 w linespoints title 'Tiled', \
     "16_460.dat" using 1:4 w linespoints title 'Textured'
     
set title "8x8 blocks"
set output "460GTX_8_blocks.png"

plot "8_460.dat" using 1:2 w linespoints title 'Simple', \
     "8_460.dat" using 1:3 w linespoints title 'Tiled', \
     "8_460.dat" using 1:4 w linespoints title 'Textured'

     
set title "4x4 blocks"
set output "460GTX_4_blocks.png"

plot "4_460.dat" using 1:2 w linespoints title 'Simple', \
     "4_460.dat" using 1:3 w linespoints title 'Tiled', \
     "4_460.dat" using 1:4 w linespoints title 'Textured'
     

set title "2x2 blocks"
set output "460GTX_2_blocks.png"

plot "2_460.dat" using 1:2 w linespoints title 'Simple', \
     "2_460.dat" using 1:3 w linespoints title 'Tiled', \
     "2_460.dat" using 1:4 w linespoints title 'Textured'




     #8800
set title "16x16 blocks"
set output "8800GTX_16_blocks.png"

plot "16_8800.dat" using 1:2 w linespoints title 'Simple', \
     "16_8800.dat" using 1:3 w linespoints title 'Tiled', \
     "16_8800.dat" using 1:4 w linespoints title 'Textured'
     
set title "8x8 blocks"
set output "8800GTX_8_blocks.png"

plot "8_8800.dat" using 1:2 w linespoints title 'Simple', \
     "8_8800.dat" using 1:3 w linespoints title 'Tiled', \
     "8_8800.dat" using 1:4 w linespoints title 'Textured'

     
set title "4x4 blocks"
set output "8800GTX_4_blocks.png"

plot "4_8800.dat" using 1:2 w linespoints title 'Simple', \
     "4_8800.dat" using 1:3 w linespoints title 'Tiled', \
     "4_8800.dat" using 1:4 w linespoints title 'Textured'
     

set title "2x2 blocks"
set output "8800GTX_2_blocks.png"

plot "2_8800.dat" using 1:2 w linespoints title 'Simple', \
     "2_8800.dat" using 1:3 w linespoints title 'Tiled', \
     "2_8800.dat" using 1:4 w linespoints title 'Textured'





     


# montage 460GTX_*.png -geometry 800x600>+2+4 samlet.png
