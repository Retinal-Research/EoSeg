# EoSeg: Does Your ViT Still Need U-Net for Segmentation?

[![Code](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/Retinal-Research/EoSeg)
[![Paper](https://img.shields.io/badge/arXiv-2607.00223-b31b1b)](https://arxiv.org/abs/2607.00223)

---

## Overview

This repository contains the official codebase for **EoSeg** and the paper:

**Does Your ViT Still Need U-Net for Segmentation?**

The project is organized around a Lightning CLI training pipeline with dataset-specific configurations for Synapse, ACDC, GlaS, ISIC, Kvasir, and MoNuSeg.

---

## Repository Structure

```text
EoSeg/
├── configs/
│   ├── acdc/
│   ├── glas/
│   ├── isic2016/
│   ├── isic2017/
│   ├── kvasir/
│   ├── monuseg/
│   └── synapse/
├── datasets/
├── models/
├── scripts/
├── training/
├── main.py
├── requirements.txt
└── README.md
```

---

## Main Entry

Training and evaluation are driven through `main.py` with Lightning CLI configs.

The primary experiment configuration is:

- `configs/synapse/vit_query_mul_scale_fusion.yaml`

The remaining Synapse configs are kept as ablation settings.

---

## Environment

This repository currently provides a single environment dependency file:

- `requirements.txt`

There is no `environment.yml` or `pyproject.toml` at the moment.

Recommended setup:

```bash
conda create -n eoseg python=3.10 -y
conda activate eoseg
pip install --upgrade pip
pip install -r requirements.txt
```

If you are running on a cluster, it is often better to install a CUDA-matched `torch` and `torchvision` first, then install the remaining packages from `requirements.txt`.

The current codebase depends on:

- `lightning` and `jsonargparse` for the CLI training pipeline
- `torch`, `torchvision`, `torchmetrics`, and `timm` for modeling and training
- `transformers` for backbone integrations
- `numpy`, `scipy`, `h5py`, `Pillow`, `matplotlib`, and `medpy` for data handling and evaluation
- `wandb` for experiment logging
- `mmengine` and `mmsegmentation` for selected backbone/util layers

---

## Training

Example training command:

```bash
python3 main.py fit --config configs/synapse/vit_query_mul_scale_fusion.yaml
```

Example evaluation command:

```bash
python3 main.py test --config configs/synapse/vit_query_mul_scale_fusion.yaml --ckpt_path /path/to/checkpoint.ckpt
```

Please update dataset paths and checkpoint paths in the YAML configs to match your local or cluster environment.

---

## Notes

- The repository keeps a compact set of visualization utilities under `scripts/`.
- The YAML files under `configs/synapse/` include the main model configuration plus retained ablation settings.

---

## Acknowledgement

This codebase is developed with code reference to:

- EOMT: [https://github.com/tue-mps/EoMT](https://github.com/tue-mps/EoMT)
- TransUNet: [https://github.com/Beckschen/TransUNet](https://github.com/Beckschen/TransUNet)

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@article{li2026does,
  title   = {Does Your ViT Still Need U-Net for Segmentation?},
  author  = {Li, Xin and Zhu, Wenhui and Dong, Xuanzhao and Chen, Xiwen and Chen, Yanxi and Xiong, Yujian and Wang, Hao and Dumitrascu, Oana M and Wang, Yalin},
  journal = {arXiv preprint arXiv:2607.00223},
  year    = {2026}
}
```
