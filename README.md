# 2LTTA
Official code release for the paper  
**Two-Level Test-Time Adaptation in Multimodal Learning**  
(ICML 2024 FM-Wild Workshop; IJCNN 2025)

## Benchmarks

We conduct experiments on the **VGGSound** and **Kinetics** datasets.
Corruptions for both video and audio modalities are generated following the
implementation provided in the **READ** repository
([GitHub](https://github.com/XLearning-SCU/2024-ICLR-READ)).
All datasets should be placed in the `data_path` directory under the project root.

Pretrained models for VGGSound and KS50 are available in the READ repository
([GitHub](https://github.com/XLearning-SCU/2024-ICLR-READ)).



## Reproduce
python run_read.py --dataset 'ks50' --json-root 'code_path/json_csv_files/ks50' --label-csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth' --tta-method '2LTTA' --severity-start 5 --severity-end 5 --corruption-modality 'video'

## Enviroments
python== 3.9
torch==1.13.1
torchaudio==0.13.1
timm==0.6.5
scikit-learn==0.22.1
numpy==1.21.6

## Citation
If you find **2LTTA** useful for your research, please cite the following papers.

### IJCNN 2025 (Extended Version – Preferred)
```bibtex
@inproceedings{lei2025twolevel,
  author    = {Lei, Jixiang and Pernkopf, Franz},
  title     = {Two-Level Test-Time Adaptation in Multimodal Learning},
  booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  address   = {Rome, Italy},
  pages     = {1--8},
  doi       = {10.1109/IJCNN64981.2025.11228216}
}


## Acknowledgement

This repository is based on and derived from the implementation accompanying
the paper **READ** (https://github.com/XLearning-SCU/2024-ICLR-READ) and **TENT** (https://github.com/DequanWang/tent). The original code is licensed under the Apache License 2.0.

