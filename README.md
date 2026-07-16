# GenDisAug

Reproduction of ["Generative or Discriminative? Revisiting Text Classification in the Era of Transformers"](https://arxiv.org/abs/2506.12181) (EMNLP 2025 Outstanding Paper) with EDA and Back-Translation data augmentation.

## Setup

conda create -n gendisc python=3.11 -y
conda activate gendisc
pip install torch transformers datasets scikit-learn pytorch-lightning nltk sentencepiece

## Run

python run_experiments.py --dataset sst2 --models bert_encoder,gpt2_ar --sample_sizes 128,512,2048,-1 --seeds 42,123,456
python run_experiments.py --dataset sst2 --models bert_encoder,gpt2_ar --sample_sizes 128,512 --seeds 42,123,456 --augment eda
python run_experiments.py --dataset sst2 --models bert_encoder,gpt2_ar --sample_sizes 128,512 --seeds 42,123,456 --augment backtrans

## Key Findings

- Pretrained BERT dominates GPT-2 at all data scales (no cross-over)
- Back-Translation outperforms EDA for low-data augmentation
- Augmentation reduces GPT-2 training variance by 3.6--6.9×

## Paper

paper/acl_paper_selfcontained.tex (Overleaf-ready)

## Reference

Kasa et al. "Generative or Discriminative?" EMNLP 2025.
Original code: [amazon-science/Generative-vs-Discriminative-Classifiers](https://github.com/amazon-science/Generative-vs-Discriminative-Classifiers)
