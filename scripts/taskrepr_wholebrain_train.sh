source venv/bin/activate

TASKS="autoencoder \
class_1000 \
class_places \
colorization \
curvature \
denoise \
edge2d \
edge3d \
inpainting_whole \
jigsaw \
keypoint2d \
keypoint3d \
reshade \
rgb2depth \
rgb2mist \
rgb2sfnorm \
room_layout \
segment25d \
segment2d \
segmentsemantic \
vanishing_point"

cd code
sub=$1
for task in $TASKS; do
    echo "running taskonomy $task task on subject $sub"
    python run_modeling.py --model taskrepr_$task --whole_brain --subj $sub --fix_testing --notest
done
