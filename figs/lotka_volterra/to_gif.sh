#! /bin/bash

R=1

rm -f *.png 
rm -f *.gif


FILES=`ls *.pdf`
for file in $FILES; do 
    filename="${file%.*}"
    inkscape $file -e $filename".png" -y 1.0
done

#ffmpeg -r $R -i oscillator_phase_space-%01d.jpg phase_space.gif 
#ffmpeg -r $R -i oscillator_output-%01d.jpg output.gif
convert -delay 120 -loop 0 *_phase_space*.png phase_space.gif
convert -delay 120 -loop 0 *out*.png output.gif



