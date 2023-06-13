# Quantization of Convolutional Neural Networks in HAR

This repository contains experiments and a framework for quantization of neural networks with PyTorch. Most results can be found in [this notebook](eval_notebooks/ptq_cnn-imu_eval.ipynb).

## Directories 

- The `common` directory contains common code used by many experiments
- The `train_scripts` directory contains all experiments. They only require 
a [Sacred](https://sacred.readthedocs.io/en/stable/index.html) MongoDB for metric logging.
- The `eval_notebook` directory contains the analysis of all logged metrics produced by the experiments.

## Thesis

All experiments were part of my bachelor thesis
[Quantisierung von Convolutional Neural Networks für die Aktivitätserkennung [German, PDF]](https://web.patrec.cs.tu-dortmund.de/pubs/theses/ba_roeger.pdf).