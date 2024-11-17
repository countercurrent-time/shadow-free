export CUDA_VISIBLE_DEVICES=3
MODE=victim 

MASTER_PORT=748857 # modify
SURROGATE_MODEL_NAME=CodeGPT-small-java
SAMPLE_RATIO=20
LANG=java                       

for temperature in 0.1 0.5 2.0
do
DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/microsoft/CodeGPT-small-java-adaptedGPT2/100/checkpoint-epoch-4"
LOGFILE="completion_javaCorpus_eval_${SURROGATE_MODEL_NAME}_${SAMPLE_RATIO}_victim.log"

python -u run.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$DATADIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --MASTER_PORT $MASTER_PORT \
        --per_gpu_eval_batch_size 128 \
        --mode $MODE \
        --temperature $temperature \
        --generate_method top-k 
done