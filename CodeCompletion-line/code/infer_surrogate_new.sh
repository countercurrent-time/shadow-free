# run codegpt infer
# run new prepare or load existing dataset first
export CUDA_VISIBLE_DEVICES=0,1
export MODE=surrogate
LANG=java
SAMPLE_RATIO=20
Percentage=0.01
DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
# DATADIR=../../CodeCompletion-token/dataset/javaCorpus/token_completion
LITFILE=../../CodeCompletion-token/dataset/javaCorpus/literals.json
# OUTPUTDIR=../../CodeCompletion-token/save/javaCorpus
PRETRAINDIR=../../CodeCompletion-token/save/javaCorpus/microsoft/CodeGPT-small-java/20/checkpoint-epoch-4
LOGFILE=completion_javaCorpus_eval.log
python -u run_lm.py \
        --mode=$MODE \
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
        --seed=42 