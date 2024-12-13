# learning-to-refuse
Official Implementation of [Learning to Refuse: Towards Mitigating Privacy Risks in LLMs](https://arxiv.org/abs/2407.10058)

## RETURN: Real-world pErsonal daTa UnleaRNing dataset

RETURN is avaliable in `data/RETURN.jsonl`. You also can access RETURN directly on [Hugging Face](https://huggingface.co/datasets/zhliu/RETURN).

```python
from datasets import load_dataset

dataset = load_dataset("zhliu/RETURN")
```

## Reproduction

### Environment Setup

```shell
# Clone the repository
git clone git@github.com:zhliu0106/learning-to-refuse.git
cd learning-to-refuse

# Create and activate conda environment
conda create -n refuse python==3.10
conda activate refuse

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing

```shell
bash scripts/data_process.sh
```

### Training and Evaluation

```shell
bash scripts/run.sh
```

## Citation

```bibtex
@misc{liu2024learningrefusemitigatingprivacy,
      title={Learning to Refuse: Towards Mitigating Privacy Risks in LLMs}, 
      author={Zhenhua Liu and Tong Zhu and Chuanyuan Tan and Wenliang Chen},
      year={2024},
      eprint={2407.10058},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10058}, 
}
```