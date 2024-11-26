export ANSWER_PATH=./answer
mkdir -p ${ANSWER_PATH}
python MIRROR.py \
    --answer_dir ${ANSWER_PATH} \
    --model_name gpt-4o-mini-2024-07-18 \
    --num_process 1 \
    --toolbench_key  "" \
    --test_set G2_instruction 