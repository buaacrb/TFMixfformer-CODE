# TFMixformer: Enhancing Time-Frequency Representation in Transformers for Long-Term Series Forecasting(Source CODE)

This repository contains the implementation of the TFMIXFORMER model along with the necessary scripts and utilities to run experiments and manage data. Below is the structure of the repository and descriptions for each directory:

- **data_provider**: Contains modules for data preprocessing and loading.
- **dataset**: Includes the datasets used for training and evaluating the model.
- **layers**: Contains core modules of the model such as attention mechanisms and other computational layers.
- **models**: Includes the complete model code integrating various layers and functionalities.
- **scripts**: Houses scripts for running experiments, including setup and execution scripts.
- **utils**: Utilities and helper functions used across the project.
- **requirements.txt**: Lists all the dependencies required to run the project.
- **run.py**: The entry point script to execute the model training and evaluation.


## Get Started

1. Install dependencies from requirements.txt.
2. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by running the following shell code separately:

```bash
bash ./scripts/ali/run_M.sh
bash ./scripts/ali/run_S.sh
bash ./scripts/elcgrid/run_M.sh
bash ./scripts/elcgrid/run_S.sh
```
