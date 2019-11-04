#!/bin/sh
cd /home/yuanw3/taskonomy/taskbank
source ~/taskonomy/venv/bin/activate
set -eu

STIMULI_DIR="/home/yuanw3/ObjectEmbeddingSpace/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/"
OUT_DIR="/home/yuanw3/ObjectEmbeddingSpace/genStimuli/"

IMAGE_DATASET="COCO \
	ImageNet \
	Scene"

TASKS="class_1000 \
class_places \
autoencoder \
denoise \
segment25d \
segment2d \
curvature \
edge2d \
edge3d \
keypoint2d \
keypoint3d \
reshade \
rgb2depth \
rgb2mist \
rgb2sfnorm \
colorization \
room_layout \
segmentsemantic \
vanishing_point \
jigsaw \
inpainting_whole"

# TASKS="ego_motion \
# fix_pose"

# TASKS="non_fixated_pose \
# point_match"

n=0
for DIR in $IMAGE_DATASET; do
	# imgtype=jpg
	# if [ $DIR == "ImageNet" ] ; then
	#	imgtype=JPEG
	# fi
	echo "$DIR"
	for imgfile in $(ls -1 $STIMULI_DIR$DIR/* | sort -r); do
	# for imgfile in $STIMULI_DIR$DIR/*; do
		n=$((n + 1))
		for task in $TASKS; do
		    store_name=$(basename $imgfile)
		    target_DIR=${OUT_DIR}${task}
		    if ! [ -e $target_DIR ]; then
			mkdir $target_DIR
		    fi
        	# echo "processing $imgfile for task $task"
        	if [ ! -e $target_DIR/$store_name ]; then
			python /home/yuanw3/taskonomy/taskbank/tools/run_img_task.py --task $task --img $imgfile --store "$target_DIR/$store_name" --store-rep
		fi
		done
	done
done
echo $n
