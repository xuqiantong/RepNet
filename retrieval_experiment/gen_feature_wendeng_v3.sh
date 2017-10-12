python gen_v3_feature.py \
    --gpu_id 0  \
    --img_dir  /media/megatron-home/dwliang/data/wendeng_res/ \
    --img_list /home/xqt/bishe/demo/retrieval_test/wendeng.txt \
    --feature_path /home/xqt/features/_v4_wendeng_v3.fea \
    --net_def /home/xqt/bishe/exp/deploy_multi_v4.prototxt \
    --weights /home/xqt/exp_res/_v4_iter_21644.caffemodel  \
    --feat  fc7 \
    --fc_model  fc8_class \
    --fc_color  fc8_color
