# 1. 把数据进行划分并保存 dataset/model/sample_ratio/
# 2. 在1:1上进行训练 (name: bert_${model}_${sample_ratio}.pth),一共保存4x3个
# 3. 在1:1上进行评测
PER_NODE_GPU=2
export CUDA_VISIBLE_DEVICES=0,1
MODEL=microsoft/CodeGPT-small-java #change model
MASTER_PORT=95497 # modify
SURROGATE_MODEL=microsoft/CodeGPT-small-java # modify:[microsoft/CodeGPT-small-java-adaptedGPT2,gpt2,microsoft/CodeGPT-small-java,rnn,transformer]
Percentage=0.01


# for SAMPLE_RATIO in {30..30..10}
for SAMPLE_RATIO in {20..20..10}
do

LANG=java    
CLASSIFIER_SAVE_DICT=../classifier_save/javaCorpus/${SURROGATE_MODEL##*/}/${SAMPLE_RATIO}/
PREDICTION_DATA_FOLDER_PATH=../../CodeCompletion-token/dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/
LITFILE=../../CodeCompletion-token/dataset/javaCorpus/literals.json

python run.py \
    --do_lower_case \
    --lang ${LANG} \
    --surrogate_model ${SURROGATE_MODEL} \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prediction_data_folder_path ${PREDICTION_DATA_FOLDER_PATH} \
    --lit_file ${LITFILE} \
    --classifier_model_path ${MODEL} \
    --weight_decay=0.01 \
    --mode checkpoint-epoch-5_surrogate \
    --seed 43

done

