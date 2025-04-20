# NLP-Deep-Learning-Transformers

# NLP & Deep Learning: Self-attention, Transformers, Pretraining

## Overview
This repository uses **Python** and **PyTorch** to apply NLP and deep learning techniques with Transformers. It explores span-corruption pretraining, self-attention mechanisms, and fine-tuning for knowledge retrieval tasks, demonstrating skills in model design, scalable workflows, and evaluation.

## Key Features & Highlights
- **Span-Corruption Pretraining**: Implemented the `CharCorruptionDataset` to mask contiguous character spans and train the model to reconstruct missing content during pretraining.
- **Knowledge-Intensive Fine-Tuning**: Automated pipelines to fine-tune a Transformer from scratch and from pretrained checkpoints on a name-to-birthplace prediction task.
- **Rotary Positional Embeddings (RoPE)**: Integrated RoPE in `attention.py` to capture relative positional information and improve model generalization.
- **Scalable Training Workflow**: Developed `run.sh` scripts and helper modules to seamlessly orchestrate pretraining, fine-tuning, and evaluation on both local and cloud environments.
- **Modular Codebase**: Structured code into clear components—data loading (`dataset.py`), model definitions (`model.py`), attention mechanisms (`attention.py`), and workflow utilities (`helper.py`).

## Results
| Experiment                             | Dev Accuracy | Improvement (Dev) |
|----------------------------------------|-------------:|-----------------:|
| No Pretraining (scratch)               |          1.2% | N/A              |
| Vanilla Pretraining + Fine-Tuning      |         26.2% | +25.0 pp         |
| RoPE Pretraining + Fine-Tuning         |         20.6% | +19.4 pp         |

## Project Structure
- **src/submission/dataset.py**: Span-corruption and name/place data definitions  
- **src/submission/model.py**: Transformer architecture implementations  
- **src/submission/attention.py**: Self-attention and RoPE functions  
- **src/submission/helper.py**: Training, pretraining, and evaluation workflows  
- **run.sh**: One-line commands to execute pretraining, fine-tuning, and evaluation pipelines  

## Insights
- **Pretraining Boosts Knowledge Retrieval**: Span-corruption pretraining produced a dramatic 25 percentage‑point jump, highlighting unsupervised pretraining’s role in encoding world knowledge.
- **RoPE Trade-offs**: Rotary embeddings enhanced relative position modeling at the cost of a ~5 pp drop versus vanilla pretraining, indicating potential for further tuning or hybrid approaches.
- **Scalability & Reproducibility**: Modular scripts and clear workflow orchestration ensure experiments are easily reproducible and extendable, facilitating rapid iteration and deployment readiness.

## Assignment Requirements
- **Vanilla Model**: Required ≥10% development accuracy; achieved 26.2% Dev.
- **RoPE Model**: Required ≥20% development accuracy; achieved 20.6% Dev.


