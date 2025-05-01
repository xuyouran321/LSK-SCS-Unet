echo $1
export CUDA_VISIBLE_DEVICES=$1
CUDA_VISIBLE_DEVICES=1 python train_supervision.py -c GeoSeg/config/loveda/mambaunet.py | tee -a mambaunet.txt
# CUDA_VISIBLE_DEVICES=1 python uavid_test.py -c GeoSeg/config/loveda/mambaunet.py -o fig_results/loveda/mamba --val --rgb -t 'd4' | tee -a output_loveda_test.txt

