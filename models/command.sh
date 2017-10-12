./build/tools/caffe train --solver=/media/optimus-home/xqt/new_exp/solver11.prototxt --snapshot=/media/optimus-home/xqt/exp_res/11_iter_120000.solverstate --gpu=15
./build/tools/caffe train --solver=/media/optimus-home/xqt/new_exp/solver12.prototxt --snapshot=/media/optimus-home/xqt/exp_res/12_iter_294000.solverstate --gpu=8
./build/tools/caffe train --solver=/media/optimus-home/xqt/new_exp/solver10.prototxt --snapshot=/media/optimus-home/xqt/exp_res/10_iter_54000.solverstate --gpu=14


./build/tools/caffe train --solver=/home/xqt/exp/solver_5.prototxt --weights=/home/xqt/exp_res/v6_iter_280000.caffemodel --gpu=0


./build/tools/caffe train --solver=/home/xqt/exp/solver_1.prototxt --weights=/home/xqt/exp_res/_v4_iter_21644.caffemodel --gpu=0

./build/tools/caffe train --solver=/home/xqt/exp/solver_3.prototxt --weights=/home/xqt/essence/v2_cmp_iter_203510.caffemodel --gpu=1

./build/tools/caffe train --solver=/home/xqt/demo/solver_clf_vgg.prototxt --gpu=1
./build/tools/caffe train --solver=/home/xqt/demo/solver_clf_vgg.prototxt --snapshot=/home/xqt/caffe/examples/try_iter_5456.solverstate --gpu=0
./build/tools/caffe train --solver=/home/xqt/demo/solver_clf_vgg.prototxt --snapshot=/home/xqt/caffe/examples/try_iter_5456.solverstate --gpu=all

./build/tools/caffe test --model=/home/xqt/demo/train_val_clf_vgg.prototxt --weights=/home/xqt/caffe/examples/try_iter_5456.caffemodel --gpu=0

vim /home/xqt/demo/solver_clf_vgg.prototxt

# prototype
./build/tools/caffe train --solver=/home/xqt/exp/solver_clf_vgg_xqt.prototxt --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_no_sigm.prototxt --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_no_sigm.prototxt --snapshot=/home/xqt/exp_res/no_sigm_iter_26342.solverstate --gpu=1
./build/tools/caffe test --model=/home/xqt/exp/train_multi_no_sigm.prototxt --weights=/home/xqt/exp_res/no_sigm_iter_75000.caffemodel --gpu=0

# v0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v0.prototxt --weights=/home/xqt/bishe/essence/v0_iter_34589.caffemodel --gpu=1

./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v02.prototxt --gpu=2


# v1
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v1.prototxt --gpu=1
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v1.prototxt --weights=/home/xqt/essence/v0_iter_34589.caffemodel --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v1.prototxt --snapshot=/home/xqt/exp_res/v1_iter_103758.solverstate --gpu=1
./build/tools/caffe test --model=/home/xqt/exp/train_multi_v1.prototxt --weights=/home/xqt/exp_res/v1_iter_25242.caffemodel --gpu=1

./build/tools/caffe train --solver=/home/xqt/exp/solver_v1_t.prototxt --weights=/home/xqt/exp_res/v1_t_iter_124563.caffemodel --gpu=1
./build/tools/caffe train --solver=/home/xqt/exp/solver_v1_t.prototxt --weights=/home/xqt/essence/base_iter_75000.caffemodel --gpu=0

./build/tools/caffe train --solver=/home/xqt/exp/solver_v1.prototxt --weights=/home/xqt/exp_res/v1_iter_165000.caffemodel --gpu=0


# v2
./build/tools/caffe train --solver=/home/xqt/exp/v2/solver_multi_v2.prototxt --gpu=1
./build/tools/caffe train --solver=/home/xqt/exp/v2/solver_multi_v2.prototxt --snapshot=/home/xqt/exp_res/v2_iter_888.solverstate --gpu=0
./build/tools/caffe test --model=/home/xqt/exp/v2/train_multi_v2.prototxt --weights=/home/xqt/essence/v2-_iter_35454.caffemodel --gpu=1

./build/tools/caffe train --solver=/home/xqt/exp/solver_v2.prototxt --snapshot=/home/xqt/essence/v2_iter_148747.solverstate --gpu=2

./build/tools/caffe train --solver=/home/xqt/exp/solver_v2.1.prototxt --weights=/home/xqt/exp_res/v2.1_iter_100000.caffemodel --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_v2_cmp.prototxt --snapshot=/home/xqt/exp_res/v2_cmp_iter_156840.solverstate --gpu=2

# v2-
./build/tools/caffe train --solver=/home/xqt/exp/0402/solver_multi_v2-.prototxt --weights=/home/xqt/exp_res/v2-_iter_35454.caffemodel --gpu=2
./build/tools/caffe test --model=/home/xqt/exp/0402/train_multi_v2-.prototxt --snapshot=/home/xqt/essence/v2-_iter_3.caffemodel --gpu=1


# v3
./build/tools/caffe train --solver=/home/xqt/exp/v2/solver_multi_v3.prototxt --weights=/home/xqt/essence/v3_iter_25000.caffemodel --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/v2/solver_multi_v3.prototxt --weights=/home/xqt/essence/v2_iter_20000.caffemodel --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/v2/solver_multi_v3.prototxt --weights=/home/xqt/essence/v1_iter_134000.caffemodel --gpu=1
./build/tools/caffe test --model=/home/xqt/exp/v2/train_multi_v3.prototxt --weights=/home/xqt/exp_res/v3_iter_2363.caffemodel --gpu=1


./build/tools/caffe train --solver=/home/xqt/exp/solver_v3.prototxt --snapshot=/home/xqt/exp_res/v3_iter_60000.solverstate --gpu=0

# v4
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v4.prototxt --weights=/home/xqt/exp_res/_v4_iter_3022.caffemodel --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v4.prototxt --weights=/home/xqt/exp_res/_v4_iter_15029.caffemodel --gpu=1
./build/tools/caffe train --solver=/home/xqt/exp/v4/solver_multi_v4.prototxt --weights=/home/xqt/exp_res/v4_iter_5921.caffemodel --gpu=1
./build/tools/caffe test --model=/home/xqt/bishe/exp/train_multi_v4.prototxt --weights=/home/xqt/exp_res/_v4_iter_103579.caffemodel --gpu=2

./build/tools/caffe train --solver=/home/xqt/bishe/exp/solver_multi_v4.prototxt --snapshot=/home/xqt/exp_res/_v4_iter_6881.solverstate --gpu=2

./build/tools/caffe train --solver=/home/xqt/exp/solver_v4.prototxt --weights=/home/xqt/exp_res/v3_iter_60000.caffemodel --gpu=2


# v5
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v5.prototxt --weights=/home/xqt/essence/v1_iter_25242.caffemodel --gpu=0
./build/tools/caffe test --model=/home/xqt/exp/train_multi_v5.prototxt --weights=/home/xqt/essence/v5_iter_27000.caffemodel --gpu=2

./build/tools/caffe train --solver=/home/xqt/exp/solver_v5.prototxt --snapshot=/home/xqt/exp_res/v5_iter_30000.solverstate --gpu=0

./build/tools/caffe train --solver=/home/xqt/exp/solver_v5.prototxt --snapshot=/home/xqt/exp_res/v5_iter_40000.solverstate --gpu=0


# v6 v6_iter_26510
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v6.prototxt --gpu=0
./build/tools/caffe train --solver=/home/xqt/exp/solver_multi_v6.prototxt --snapshot=/home/xqt/exp_res/v5_iter_1118.solverstate --gpu=2

./build/tools/caffe train --solver=/home/xqt/exp/solver_v6.prototxt --weights=/home/xqt/essence/base2_iter_4000.caffemodel --gpu=2
./build/tools/caffe train --solver=/home/xqt/exp/solver_v6.prototxt --snapshot=/home/xqt/exp_res/v6_iter_100000.solverstate --gpu=2

# v7 
./build/tools/caffe train --solver=/home/xqt/exp/solver_pre_v7.prototxt --snapshot=/home/xqt/exp_res/v7_iter_131898.solverstate --gpu=2



