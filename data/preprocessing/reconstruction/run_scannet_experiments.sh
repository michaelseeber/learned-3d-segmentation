SCANNET_PATH=/scratch/thesis/data

for scene_path in $(ls -d $SCANNET_PATH/scenes/reconstruct_gt/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    if [ ! -d ${scene_path}/label ]; then
        unzip $scene_path/${scene_name}_2d-label.zip -d ${scene_path}
    fi

    if [ ! -d ${scene_path}/sensor ]; then
        $SCANNET_PATH/preprocessing/reconstruction/SensReader/c++/sens \
            $scene_path/${scene_name}.sens $scene_path/sensor
    fi

    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images
    mkdir -p $scene_path/converted/groundtruth_model

    # Convert scannet data into a format read by our code
    python3 $SCANNET_PATH/preprocessing/reconstruction/TSDF/convert_scannet.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --resolution 0.05

    # Fuse all depth maps and segmentation to create the ground-truth
    python $SCANNET_PATH/preprocessing/reconstruction/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/groundtruth_datacost \
        --resolution 0.05

    # Run total variation on the datacost obtained from all depth
    # to generate the ground truth voxel grid
    python $SCANNET_PATH/preprocessing/reconstruction/eval_tv_l1.py \
        --datacost_path $scene_path/converted/groundtruth_datacost.npz \
        --output_path $scene_path/converted/groundtruth_model \
        --label_map_path $scene_path/converted/labels.txt \
        --niter_steps 50 --lam 0.1 --nclasses 22

done

