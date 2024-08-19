# Exploring Robustness of GNN against Universal Injection Attack from a Worst-case Perspective

A PyTorch implementation of "Exploring Robustness of GNN against Universal Injection Attack from a Worst-case Perspective" (CIKM 2024).

## Introduction
We propose a method named **CERT_UIA** to enhance the robustness of GNN models against worst-case attacks, specifically targeting the scenario of **U**niversal node **I**njection **A**ttacks (UIA), thereby filling a gap in the existing literature on certified robustness in this context. Our approach involves a two-stage attack process that replaces the transformations of the topology and feature spaces with equivalent unified feature transformations, unifying the optimization of worst-case perturbations into a single feature space.

## Cert_UIA
![Overview of Cert_UIA](https://github.com/Eve-Ni/Cert_UIA/blob/master/model.png "Overview of Cert_UIA")

## Requirements
* Python 3.9
* PyTorch 2.3.0
* torch_geometric

## Datasets
[TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)
* PROTEINS
* IMDB-BINARY
* REDDIT-BINARY

## Usage
**Configure**
```python
cd configs/
```
**Run**
```python
python main.py
```

## Cite
```python
@inproceedings{
  title={Exploring Robustness of GNN against Universal Injection Attack from a Worst-case Perspective},
  author={Dandan Ni and Sheng Zhang and Cong Deng and Han Liu and Gang Chen and Minhao Cheng and Hongyang Chen.},
  booktitle={Proceedings of the 33th ACM International Conference on Information and Knowledge Management},
  series={CIKM'24},
  year={2024}
}
```
