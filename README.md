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
@article{liu2024learning,
  title={Learning to refuse: Towards mitigating privacy risks in llms},
  author={Liu, Zhenhua and Zhu, Tong and Tan, Chuanyuan and Chen, Wenliang},
  journal={arXiv preprint arXiv:2407.10058},
  year={2024}
}
```