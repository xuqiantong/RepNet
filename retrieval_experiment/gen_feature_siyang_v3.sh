python gen_v3_feature.py \
    --gpu_id 0  \
    --img_dir /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --img_list /home/xqt/exp/siyang.txt \
    --feature_path /home/xqt/features/_v4_siyang_v3.fea \
    --net_def /home/xqt/bishe/exp/deploy_multi_v4.prototxt\
    --weights  /home/xqt/exp_res/_v4_iter_21644.caffemodel \
    --feat  fc7 \
    --fc_model  fc8_class \
    --fc_color  fc8_color