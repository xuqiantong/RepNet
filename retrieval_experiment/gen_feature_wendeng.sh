python gen_v1_feature.py \
    --gpu_id 1  \
    --img_dir  /media/megatron-home/dwliang/data/wendeng_res/ \
    --img_list /home/xqt/bishe/demo/retrieval_test/wendeng.txt \
    --feature_path /home/xqt/features/v2_cmp_wendeng_tcc.fea \
    --net_def /home/xqt/exp/deploy_v2_cmp.prototxt \
    --weights /home/xqt/exp_res/v2_cmp_iter_126483.caffemodel  \
    --feat  fc9_triplet \
    --fc_model  fc8_class \
    --fc_color  fc8_color 
