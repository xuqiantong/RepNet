python search_v1.py \
    --gpu_id 1 \
    --query_dir  /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --query_list /home/xqt/exp/query_3.txt \
    --siyang_label /home/xqt/exp/siyang_label.txt \
    --net_def /home/xqt/exp/deploy_v2_cmp.prototxt \
    --weights /home/xqt/exp_res/v2_cmp_iter_126483.caffemodel \
    --feat fc9_triplet \
    --fc_model  fc8_class \
    --fc_color  fc8_color \
    --begin_loc 3 \
    --end_loc 303 \
    --siyang_feat /home/xqt/features/v2_cmp_siyang_tcc.fea \
    --wendeng_feat /home/xqt/features/v2_cmp_wendeng_tcc.fea \
    --mAP_path  /home/xqt/essence/V2_cmp_3_map.txt \
    --p@k_path  /home/xqt/essence/V2_cmp_3_precision_at_k.txt

