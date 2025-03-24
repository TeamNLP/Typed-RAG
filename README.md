# <div align="center">Typed-RAG</div>

<div align="center">
<a href="https://arxiv.org/abs/2503.15879" target="_blank"><img src=https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/oneonlee/Wiki-NFQA" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Datasets-yellow.svg></a>
<a href="https://github.com/TeamNLP/Typed-RAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg"></a>
<a href="https://www.python.org/downloads/release/python-390" target="_blank"><img src=https://img.shields.io/badge/Python-3.9-blue.svg></a>
</div>


## 📣 Latest News
- **03/22/2025**: The Wiki-NFQA dataset is released at [Hugging Face Datasets](https://huggingface.co/datasets/oneonlee/Wiki-NFQA).
- **03/20/2025**: The paper and code for Typed-RAG is available. You can access the paper on [arXiv](https://arxiv.org/abs/2503.15879) and [HF-paper](https://huggingface.co/papers/2503.15879).
- **03/13/2025**: The paper for Typed-RAG is accepted at [NAACL 2025 SRW](https://naacl2025-srw.github.io/accepted)!

## 💡 Overview
Typed-RAG enhances retrieval-augmented generation for non-factoid question-answering (NFQA) through type-aware multi-aspect query decomposition, delivering more contextually relevant and comprehensive responses.

## 📚 Wiki-NFQA Dataset
The Wiki-NFQA Dataset is a curated benchmark designed for evaluating open-domain question answering (ODQA) systems with non-factoid questions. The dataset is available on Hugging Face.

```python
from datasets import load_dataset

# Load the combined dataset with all examples
wiki_nfqa_dataset = load_dataset("oneonlee/Wiki-NFQA", "Wiki-NFQA", split="test")

# Load reference answers for evaluation
reference_answers = load_dataset("oneonlee/Wiki-NFQA", "reference_answer_list", split="test")
```

## 🔧 Usage
### Installation
```bash
conda create -n nfqa python=3.9
conda activate nfqa
pip install -r requirements.txt
```

### Dataset Preparation & Preprocessing
```bash
sh scripts/download_raw_data.sh
sh scripts/process_data.sh
sh scripts/classify_data.sh
sh scripts/sample_data.sh
```

### Elasticsearch Setup
```bash
python retriever/process_wiki.py
sh scripts/elasticsearch_setup.sh
```

### Reference Answers Construction & Annotation
```bash
sh scripts/construct_reference_list.sh
```

### Experiments
```bash
# Should run ES retriever manually
nohup ./retriever/elasticsearch-7.9.1/bin/elasticsearch > elasticsearch.log &
```

```bash
sh scripts/run_baseline_hf.sh
sh scripts/run_baseline_gpt.sh
sh scripts/run_Typed-RAG_hf.sh
sh scripts/run_Typed-RAG_gpt.sh
```

### Evaluation
```bash
sh scripts/LINKAGE_mistral.sh
sh scripts/LINKAGE_gpt.sh
```

```bash
python evaluation/evaluate_MRR_MPR.py
python evaluation/extract_results.py
python evaluation/concat_prediction_files.py
```

## 📄 Citation
```bib
@misc{lee2025typedrag,
      title={Typed-RAG: Type-aware Multi-Aspect Decomposition for Non-Factoid Question Answering}, 
      author={DongGeon Lee and Ahjeong Park and Hyeri Lee and Hyeonseo Nam and Yunho Maeng},
      year={2025},
      eprint={2503.15879},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15879}, 
}
```

## 📄 License
This project is licensed under the [CC BY-SA 4.0 license](https://github.com/TeamNLP/Typed-RAG/blob/main/LICENSE).
