# CCD
This repo contains the Pytorch implementation of our paper: 
> [**Constrained Contrastive Distribution Learning for Unsupervised Anomaly Detection and Localisation in Medical Images**](https://arxiv.org/pdf/2103.03423.pdf)
>
> [Yu Tian](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), Fengbei Liu, Seon Ho Shin, Johan W Verjans, Rajvinder Singh, [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at MICCAI 2021.**  


## Dataset

Please download the **Hyper-Kvasir Anomaly Detection** Dataset from this [**link**](https://drive.google.com/file/d/1-D4noq1V115JXL_uxz-h-P7PNZEP4j1y/view?usp=sharing). 


## Training
The code is build based on the [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification).




Modify the dataloader (data/lag_loader.py) code for your own medical images, then simply run the following command: 
```shell
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```


## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{tian2021constrained,
  title={Constrained contrastive distribution learning for unsupervised anomaly detection and localisation in medical images},
  author={Tian, Yu and Pang, Guansong and Liu, Fengbei and Chen, Yuanhong and Shin, Seon Ho and Verjans, Johan W and Singh, Rajvinder and Carneiro, Gustavo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={128--140},
  year={2021},
  organization={Springer}
}

```
---




