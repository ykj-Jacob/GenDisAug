# GenDisAug

Reproduction of ["Generative or Discriminative?"](https://arxiv.org/abs/2506.12181) (EMNLP 2025 Outstanding Paper) with EDA and Back-Translation augmentation.

Uses the **standard SST-2 split** (6,920/872) matching the original paper.

## Setup

conda create -n gendisc python=3.11 -y
conda activate gendisc
pip install torch transformers datasets scikit-learn pytorch-lightning nltk sentencepiece

## Run

# Baseline reproduction
python run_experiments.py --dataset sst2_orig --models bert_encoder,gpt2_ar --sample_sizes 128,512,2048,-1 --seeds 42,123,456

# EDA augmentation
python run_experiments.py --dataset sst2_orig --models bert_encoder,gpt2_ar --sample_sizes 128,512 --seeds 42,123,456 --augment eda

# Back-Translation augmentation
python run_experiments.py --dataset sst2_orig --models bert_encoder,gpt2_ar --sample_sizes 128,512 --seeds 42,123,456 --augment backtrans

## Results (SST-2 standard split, 3 seeds)

### Baseline

| Model | K=128 | K=512 | K=2048 | K=Full |
|-------|-------|-------|--------|--------|
| BERT | 79.6 | 86.0 | 87.2 | 90.2 |
| GPT-2 | 48.2 | 58.6 | 52.0 | 51.2 |

### Augmentation

| Model | Aug | K=128 | K=512 |
|-------|-----|-------|-------|
| BERT | EDA | 77.5 | 86.1 |
| BERT | BackTr | 78.9 | 82.4 |
| GPT-2 | EDA | 52.5 | 60.3 |
| GPT-2 | BackTr | 52.7 | 59.7 |

## Key Findings

- Pretrained BERT dominates GPT-2 at all data scales (no cross-over)
- Augmentation modestly helps GPT-2 (+1–5%), mixed effects on BERT
- EDA and Back-Translation perform comparably on standard SST-2
- Augmentation effectiveness depends on underlying dataset scale

## Paper

paper/acl_paper_selfcontained.tex (Overleaf-ready)

## Reference

Kasa et al. "Generative or Discriminative?" EMNLP 2025.
Original code: [amazon-science/Generative-vs-Discriminative-Classifiers](https://github.com/amazon-science/Generative-vs-Discriminative-Classifiers)
