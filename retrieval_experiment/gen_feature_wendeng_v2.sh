python gen_v1_feature.py \
    --gpu_id 1  \
    --img_dir  /media/megatron-home/dwliang/data/wendeng_res/ \
    --img_list /home/xqt/bishe/demo/retrieval_test/wendeng.txt \
    --feature_path /home/xqt/features/v2_wendeng_2348.fea \
    --net_def /home/xqt/exp/deploy_v2.prototxt \
    --weights /home/xqt/exp_res/v2_iter_212351.caffemodel  \
    --feat  fc9_triplet \
    --feat_cc fc7_cc
