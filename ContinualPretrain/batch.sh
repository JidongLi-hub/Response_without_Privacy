# 产生80K之前得1ksteps的训练数据，用他们训练后得到opertimizer
# python3 ../pythia/utils/batch_viewer.py \
#   --start_iteration 79000 \
#   --end_iteration 80000 \
#   --load_path /data/fangly/ljd/dataset/pile-deduped-pythia-preshuffled-Merged/document.idx \
#   --save_path data/ \
#   --output_file 79k-80k-steps-again.npy \
#   --conf_dir ../pythia/utils/dummy_config.yml 
#  产生80K之后的需要被插入新数据的训练数据，可能需要reshape以适应设备需求，根据插入频率计算需要多少条数据
python3 ../pythia/utils/batch_viewer.py \
  --start_iteration 80000 \
  --end_iteration 81000 \
  --load_path /data/fangly/ljd/dataset/pile-deduped-pythia-preshuffled-Merged/document.idx \
  --save_path data/ \
  --output_file 80k-81k-steps.npy \
  --conf_dir ../pythia/utils/dummy_config.yml 