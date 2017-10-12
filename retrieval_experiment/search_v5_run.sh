python search_v3.py \
    --gpu_id 1 \
    --query_dir  /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --query_list /home/xqt/exp/query_3.txt \
    --siyang_label /home/xqt/exp/siyang_label.txt \
    --net_def /home/xqt/exp/deploy_v5.prototxt \
    --weights /home/xqt/essence/v5_iter_30306.caffemodel \
    --feat fc9_triplet \
    --fc_model  fc8_class \
    --fc_color  fc8_color \
    --begin_loc 258 \
    --end_loc 558 \
    --siyang_feat /media/mmr6-raid5/xqt/features/v5_siyang_2348.fea \
    --wendeng_feat /media/mmr6-raid5/xqt/features/v5_wendeng_2348.fea \
    --mAP_path  /home/xqt/essence/v5_bucket_3_map.txt \
    --p@k_path  /home/xqt/essence/v5_bucket_3_precision_at_k.txt

