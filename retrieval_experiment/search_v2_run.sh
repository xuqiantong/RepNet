python search_v2.py \
    --gpu_id 1 \
    --query_dir  /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --query_list /home/xqt/exp/query_3.txt \
    --siyang_label /home/xqt/exp/siyang_label.txt \
    --net_def /home/xqt/exp/deploy_v2_cmp.prototxt \
    --weights /home/xqt/essence/v2_cmp_iter_203510.caffemodel \
    --feat fc9_triplet \
    --feat_cc fc7_cc \
    --begin_loc 258 \
    --end_loc 2606 \
    --siyang_feat /media/mmr6-raid5/xqt/features/v2_cmp_siyang_v3.fea \
    --wendeng_feat /media/mmr6-raid5/xqt/features/v2_cmp_wendeng_v3.fea \
    --mAP_path  /home/xqt/essence/v2_cmp_2348_3_map.txt \
    --p@k_path  /home/xqt/essence/v2_cmp_2348_3_precision_at_k.txt

