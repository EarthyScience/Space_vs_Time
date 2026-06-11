# Repository to the paper: Space Beats Time in Training Machine Learning Models for Carbon Uptake Variability

## Authors
[David Hafezi Rachti](https://www.bgc-jena.mpg.de/person/124768/2200), [Alexander J. Winkler](https://www.bgc-jena.mpg.de/person/121850/2200), and [Christian Reimers](https://www.bgc-jena.mpg.de/person/121686/2200christian-profile)

*Max Planck Institute for Biogeochemistry, Jena, Germany*

## Abstract
Machine learning (ML) is widely used to upscale in-situ ecosystem carbon flux observations to the globe. While performing well on seasonal cycles and spatial patterns, these methods notoriously fail to reproduce long-term trends and interannual variability (IAV). Whether this shortcoming stems from low signal-to-noise ratios or insufficient training data remains unclear.
Here, we examine how an interpretable ML framework responds to three progressively larger training datasets with contrasting spatial and temporal properties. Improving spatial representation in the training data increases the ability to predict trends and IAV more than extending the time series ($\Delta R^2 = 0.83$ vs. $\Delta R^2 = 0.60$ for IAV). Using interpretable ML, we attribute this improvement to better characterization of water-carbon dynamics through richer spatial sampling. Our results demonstrate that insufficient spatial coverage, not noise in the data, limits model performance, implying that expanding the spatial coverage of ecosystem observations is essential for reliable, data-driven estimates of carbon uptake variability.

## Content
```text
project/
├── data/
│   ├── pixel_information/
│   │
│   ├── datasets/
│   │   ├── basic_obs/
│   │   ├── space_set/
│   │   ├── time_set/
│   │   ├── timespace_set/
│   │   └── test_sets/
│   │
│   ├── pre_processing_steps/
│   │
│   └── treeFrac.nc
│
├── js_bach_data/
│   └── pre_processing_steps/
│
├── trained_models/
│   └── version1/
│       └── ig_results/
│           └── plots/
│       └── ig_results_anom/
│       └── results/
│           └── model_performance/
│       └── space_set/
│       └── time_set/
│       └── timespace_set/
│
├── data_preprocessing/
│
├── model_evaluation/
│
├── model_training/
│
├── logs/
│   └── slurm_output/
│
├── environment.yml
├── README.md
```

## Environment setup

To reproduce the environment used in this project, install the Conda environment:

```bash
conda env create -f environment.yml
```
