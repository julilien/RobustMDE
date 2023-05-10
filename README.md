# Robust Regression for Monocular Depth Estimation

The (Tensorflow 2.*) implementation of the superset learning-based robust depth regression method and the experimental evaluation as presented in the corresponding paper "Robust Regression for Monocular Depth Estimation" published at ACML 2021. Cite the paper as follows:

``` 
@InProceedings{pmlr-v157-lienen21a,
  title = 	 {Robust Regression for Monocular Depth Estimation},
  author =       {Lienen, Julian and Nommensen, Nils and Ewerth, Ralph and H\"ullermeier, Eyke},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {1001--1016},
  year = 	 {2021},
  editor = 	 {Balasubramanian, Vineeth N. and Tsang, Ivor},
  volume = 	 {157},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--19 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v157/lienen21a/lienen21a.pdf},
  url = 	 {https://proceedings.mlr.press/v157/lienen21a.html}
}
```

## Requirements

A detailed list of requirements can be found in `requirements.txt`. To install all required packages, one simply has to call
```
pip install -r requirements.txt
```

Note that the code has been tested with Python 3.8 on Ubuntu 20.04. Since we tried to avoid using any system-dependent call or library, we expect the code to be running also on other systems, such as Windows and MacOS. Possibly, path delimiters must be changed to run on non-unixoid systems.

In some cases, we've experienced issues with the MySQL adapter package for Python 3.* (used to store experimental results using [Mlflow](https://mlflow.org/)), for which the pip package install was not sufficient to run the code. On Linux systems, this package may require to install additional system-dependent sources (e.g., for Ubuntu, we also had to run `sudo apt install build-essential python-dev libmysqlclient-dev`).

## Repository Structure

This repository provides not only the implementation of our own robust superset learning losses, it also comprises the implementation of the baseline losses. The following list gives a more detailed overview over the individual packages that this repository contains:

- `robustdepth.data` provides data access objects and preprocessing code.
- `robustdepth.eval` contains the implementation of the final cross-dataset generalization evaluation.
- `robustdepth.experiments` comprises the entry point to run the experiments.
- `robustdepth.losses` provides the implementation of the baseline losses and our proposed methods.
- `robustdepth.metrics` includes depth-related metrics which we used within our evaluation.
- `robustdepth.models` includes the used EfficientNet-based model.
- `robustdepth.util` contains utility functions used throughout the project.

A basic configuration file is given by `conf/run.ini`. Here, Mlflow parameters, logging settings and data paths for each dataset can be specified. It is worth to note that the `CACHE_PATH_PREFIX` specifies a location to which all intermediate results should be stored (so make sure to grant enough space, e.g., to save every model within the random search to eventually gather the best model for the final assessment).

## Datasets

In order to train and assess the results, we used the following data set version:

- NYU: For training, we used the preprocessed training data as provided in the [DenseDepth repository](https://github.com/ialhashim/DenseDepth). This data must be stored in a directory `data` in the path specified in `conf/run.ini`. For testing, the Eigen split data must be provided in the directory `nyu_eigen_test_split`. This data is extracted from the originally provided labeled *.mat file using the code as provided in the [BTS repository](https://github.com/cogaplex-bts/bts/blob/master/utils/extract_official_train_test_set_from_mat.py). We omitted the scene names and stored the resulting images and depth annotations all in the same directory.
- SunRGBD: For training, we used the official [training](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip) and [test](http://rgbd.cs.princeton.edu/data/LSUN/SUNRGBDLSUNTest.zip) split as provided by the authors. One has to download and extract the two archives to the directory provided in `conf/run.ini`.
- iBims-1: We considered this data set within our hyperparameter optimization. For usage, one just has to extract the data set as provided [here](https://www.bgu.tum.de/lmf/ibims1/) to the directory configured in `conf/run.ini`.
- DIODE: The validation split can be downloaded [here](https://diode-dataset.org/). We considered the "DIODE Depth" validation set and used the indoor scenes. Thus, this data simply has to be extracted to a directory given in `conf/run.ini`.

## Model Evaluation

The repository's entry point is `robustdepth.experiments.robust_depth_regression.py`, which can be run in order to train a PL model for depth estimation. For our implementation, we used [Click](https://click.palletsprojects.com/en/7.x/) to provide a convenient CLI for passing parameters. Therefore, you can print out all possible program arguments with the `--help` parameter.

In our experiments, we conducted multiple runs per loss, dataset, number of used samples and noise levels with different seeds. To execute each of the run (e.g., for the trapezoidal FOSLL1 variant, 10k examples of NYU and a noise level of 0.5), you simply have to call the following:

```
python3 robustdepth/experiments/robust_depth_regression.py --loss_name fosll1_trapezoidal --dataset NYU --num_data_points 10000   --noise_level 0.5
```

This script will automatically perform the random search hyperparameter optimization as described in the paper and supplementary material, also taking care of selected the best among all trained models for the final result calculation. By specifying the `bo-directoy` parameter, you can set a specific directory to save the models within the run. By default, a directory `bo` will be created in the current working directory. The final results can be investigated using Mlflow. 

## Results

The exhaustive result tables can be found in the evaluation section of the paper and in the supplementary material. Due to space limitations, we refer to this for an overview. As used within our code and mentioned before, Mlflow allows to efficiently aggregate the produced results. We used this framework to track our results.