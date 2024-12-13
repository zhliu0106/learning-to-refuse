project_dir=$(pwd)
config_dir=${project_dir}/config
model=Meta-Llama-3-8B-Instruct

# all possible loss types
# [prefix] (optional): augmented
# [middle] (required): ga, idk, npo, idk_dpo, refuse, refuse_dpo
# [suffix] (optional): difference, kl_minimization

for loss_type in augmented_refuse_kl_minimization augmented_refuse_difference; do
    accelerate launch --config_file ${project_dir}/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes 4 \
        -m src.main \
        --config-path ${config_dir} \
        --config-name refuse.yaml \
        project_dir=${project_dir} \
        loss_type=${loss_type} \
        per_device_train_batch_size=8 \
        gradient_accumulation_steps=1 \


    mkdir -p ${project_dir}/results/${model}/${loss_type}
    python -m src.eval \
        --config-path ${config_dir} \
        --config-name refuse.yaml \
        project_dir=${project_dir} \
        loss_type=${loss_type}
done

find . -type d -name "__pycache__" -o -name "*.pyc" | xargs rm -rf
