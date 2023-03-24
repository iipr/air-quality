# Spatio-Temporal Air Quality Forecaster (ST-AQF)

Air quality is fundamental for living beings anywhere.
Being able to predict when and where certain air pollutants will strike allows for better regulation and action to tackle them.
This repository collects source code and information about a Deep Learning model named ST-AQF for air quality forecasting from 11 different pollutants using exogeous data.
This forecasting problem can be graphically posed as follows:

![](https://github.com/iipr/air-quality/raw/main/notebooks/graphs/the-problem.png)

The main features of the developed ST-AQF model are that:
- it works with air quality mesh-grids, which provide **high spatio-temporal resolution** (500x500m grid with 1h intervals),
- it can work with **multiple pollutants** both as input data and forecasts (up to 11 for the studied use case),
- the prediction takes into account `n_x` previous **timesteps**, and forecasts demand for `n_y` **future intervals**,
- it is trained with **extensive real-world** data from 24 sensors gathered over a decade;
- it is **sensor-agnostic**, and therefore:
  - it is **robust**, i.e., it can recover from sensor failure and continue producing reliable predictions,
  - it is **flexible**, i.e., it can work with a variable number of input and output sensors.

Raw and processed pollutant data can be found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7764528.svg)](https://doi.org/10.5281/zenodo.7764528)

The contents of the repository are described in the following points:
- `notebooks`: Jupyter notebooks that present how the data was analyzed and parsed. Besides, the following folders are included:
  - `graphs`: Relevant graphs about the problem, and several maps of the city chosen as the case of study: Madrid.
  - `variables-stats`: Statistics about the pollutants and weather variables used.
- `results`: Summary that includes metadata and metrics of the trained models and baselines. It includes the following folders with graphical experimental results:
  - `experiment-n_x`: Variation in the number of input time intervals.
  - `experiment-extensibility`: Comparison between models that work with an increasing number of pollutants.
  - `experiment-integrability`: Integration of exogenous data (weather and temporal information).
  - `experiment-robustness`: Simulation of sensors breakdown while continuing to produce reliable predictions.
- `src`: It includes the following scripts:
  - `deep_playground.py`: Interactive script to train and test deep learning models.
  - `launcher.py`: Script that launches the training and testing of models iteratively.
  - `learner.py`: Main script that works together with `modelUtils.py`, `trainUtils.py` and `plotUtils.py` to manage the training process, while storing the relevant model metadata.
  - `modelUtils.py`: It defines a class that contains the hardcoded models, alongside the relevant model (e.g. `n_x` and `n_y`) and training (e.g. loss function, learning rate, optimizer, etc.) parameters.
  - `plotUtils.py`: It defines a class for plotting tasks.
  - `test_gpu.py`: Script to test whether the GPU/s is/are available.
  - `trainUtils.py`: It defines a data generator class for the training and testing of the models, and a class for keeping the relevant information of the training history.


