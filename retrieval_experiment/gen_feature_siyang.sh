python gen_v1_feature.py \
    --gpu_id 1  \
    --img_dir /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --img_list /home/xqt/exp/siyang.txt \
    --feature_path /home/xqt/features/v2_cmp_siyang_tcc.fea \
    --net_def /home/xqt/exp/deploy_v2_cmp.prototxt\
    --weights  /home/xqt/exp_res/v2_cmp_iter_126483.caffemodel \
    --feat  fc9_triplet \
    --fc_model  fc8_class \
    --fc_color  fc8_color 