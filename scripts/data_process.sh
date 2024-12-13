# export CUDA_VISIBLE_DEVICES=6
export HYDRA_FULL_ERROR=1

project_dir=$(pwd)
config_dir=${project_dir}/config

python -m process_data.eval \
    --config-path ${config_dir} \
    --config-name process_data.yaml \
    hydra/job_logging=disabled \
    project_dir=${project_dir}

python -m process_data.filtering \
    --config-path ${config_dir} \
    --config-name process_data.yaml \
    hydra/job_logging=disabled \
    project_dir=${project_dir}

python -m process_data.construction \
    --config-path ${config_dir} \
    --config-name process_data.yaml \
    hydra/job_logging=disabled \
    project_dir=${project_dir}

python -m process_data.augmentation \
    --config-path ${config_dir} \
    --config-name process_data.yaml \
    hydra/job_logging=disabled \
    project_dir=${project_dir}

find . -type d -name "__pycache__" -o -name "*.pyc" | xargs rm -rf
