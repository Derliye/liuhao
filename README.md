# [AAAI2026] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2511.13607)

</div>

## Proposed ICLR
<details open>
<summary>

### **Motivation**

</summary>

![results3](./pic/figure1.png)
</details>

<details close>
<summary>

### **ICLR pipeline**

</summary>

![results3](./pic/figure2.png)
</details>

## Get Started

### Dependencies and Installation
- Python 3.7
- Pytorch 1.13

1. Create Conda Environment

```
conda create --name ICLR python=3.7
conda activate ICLR
```

2. Install PyTorch

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

3. Clone Repo

```
git clone https://github.com/Derliye/ICLR.git
```
4. Install Dependencies

```
cd ICLR
pip install -r requirements.txt
```

### Data Preparation

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

Then, put them in the following folder:

<details open> <summary>dataset (click to expand)</summary>

```
├── datasets
    ├── LOLdataset
        ├── our485
            ├──low
            ├──high
        ├── eval15
            ├──low
            ├──high
   ├── LOLv2
       ├── Real_captured
           ├── Train
	         ├── Test
       ├── Synthetic
           ├── Train
	         ├── Test
```

</details>
