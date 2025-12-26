# 2LTTA
Code for the paper "Two-Level Test-Time Adaptation in Multimodal Learning" (ICML 2024 FM-Wild Workshop))

## Benchmarks
We utilized the VGGSound and Kinetics datasets in our experiments. Corruptions for both video and audio modalities were generated following the implementation provided in the READ repository: https://github.com/XLearning-SCU/2024-ICLR-READ. All datasets should be placed in the "data_path" directory under the project root.

## Reprocude
python run_read.py --dataset 'ks50' --json-root 'code_path/json_csv_files/ks50' --label-csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth' --tta-method '2LTTA' --severity-start 5 --severity-end 5 --corruption-modality 'video'

## Enviroments
python== 3.9
torch==1.13.1
torchaudio==0.13.1
timm==0.6.5
scikit-learn==0.22.1
numpy==1.21.6

## Citation
If 2LTTA is useful for your research, please cite the following paper:
@inproceedings{
lei2024twolevel,
title={Two-Level Test-Time Adaptation in Multimodal Learning},
author={Jixiang Lei and Franz Pernkopf},
booktitle={ICML 2024 Workshop on Foundation Models in the Wild},
year={2024},
url={https://openreview.net/forum?id=n0lDbIKVAT}
}

## Acknowledgement

This repository is based on and derived from the implementation accompanying
the paper **READ**(https://github.com/XLearning-SCU/2024-ICLR-READ) and **TENT**(https://github.com/DequanWang/tent)
The original code is licensed under the Apache License 2.0.

