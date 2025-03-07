# Typed-RAG

## Wiki-NFQA Dataset
The Wiki-NFQA Dataset is a curated benchmark designed for evaluating open-domain question answering (ODQA) systems with non-factoid questions. This dataset is stored in the following location:
```bash
data/reference_list_construction/$subdataset/annotated_odqa_nf_test.jsonl
```


## Usage
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