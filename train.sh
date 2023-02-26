# Training parameters
num_epochs=1
batch_size=2 
lr=1e-3
weight_decay=4e-6
momentum=0.9


# Saving parameters
ckpt_save_path='./ckpts'
ckpt_prefix='cktp_epoch_'
ckpt_save_freq=10
report_path="./reports"
x_path='./data/trainSet/Stimuli'
y_path='./data/trainSet/FIXATIONMAPS'
regex_for_category="\.\/data\/trainSet\/Stimuli\/(.*)\/\d*\.jpg"




python train.py --batch-size $batch_size \
                --lr $lr \
                --weight-decay $weight_decay \
                --num-epochs $num_epochs \
                --ckpt-save-path $ckpt_save_path \
                --ckpt-prefix $ckpt_prefix \
                --ckpt-save-freq $ckpt_save_freq \
                --report-path $report_path \
                --x-path $x_path \
                --y-path $y_path \
                --regex-for-category $regex_for_category \
                --momentum $momentum
