# Unofficial implementation of phase reconstruction method with von Mises distribution DNN

This repository provides an unofficial implementation of phase reconstruction with von Mises distribution DNN [1][2].

## Licence
MIT licence.

Copyright (C) 2024 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.12`. The following modules are required:

- hydra-core
- joblib
- librosa
- numpy
- progressbar2
- pydub
- pypesq
- pystoi
- scikit-learn
- soundfile
- torch


## Datasets
You need to prepare the following two datasets from [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

   - basic5000: for training

   - onomatopee300: for evaluation

## Recipes

1. Download the two datasets. Put those in /root_dir/trainset_dir and /root_dir/evalset_dir/, respectively.

2. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

3. Run `preprocess.py`. It performs preprocessing steps.

4. Run `training.py`. It performs model training.

5. Run `evaluate_scores.py`. It generates reconstructed audio data and computes objective scores (PESQ, STOI, LSC).

6. Run `evaluate_scores_zerophase.py`. It also generates reconstructed audio data and computes objective scores (PESQ, STOI, LSC), where phase spectrum is assumed to be zero (zero-phase).

7. Run `plot_boxplot.py`. It plots boxplot of objective scores.

## References

[1] 高道 慎之介, 齋藤 佑樹, 高宗 典玄, 北村 大地, 猿渡 洋, "von Mises分布DNNに基づく振幅スペクトログラムからの位相復元," 情報処理学会研究報告, 2018-MUS-122, no.54, pp. 1--6, Jun., 2018.

[2] Shinnosuke Takamichi, Yuki Saito, Norihiro Takamune, Daichi Kitamura, and Hiroshi Saruwatari, "Phase reconstruction from amplitude spectrograms based on von-Mises-distribution deep neural network," Proc. IWAENC, pp. 286--290, Tokyo, Japan, Sep. 2018.
