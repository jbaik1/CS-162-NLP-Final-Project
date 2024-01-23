DATA_DIR="datasets/com2sense"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"
TASK_NAME="com2sense"
OUTPUT_DIR=${TASK_NAME}


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path "/home/jbaik/CS162-course-project/outputs/com2sense/ckpts/ckpts/checkpoint-1750" \
  --do_eval \
  --eval_all_checkpoints \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --max_steps 4000 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --save_steps 1000 \
  --logging_steps 1000 \
  --warmup_steps 100 \
  --eval_split "test" \
  --score_average_method "micro" \
  --do_not_load_optimizer \
  #--overwrite_output_dir \
