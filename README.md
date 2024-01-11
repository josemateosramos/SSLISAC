# Semi-Supervised End-to-End Learning for Integrated Sensing and Communications

## Getting Started
This code is based on [Pytorch](https://pytorch.org/) 1.12.1 and CUDA 11.3.1, and may not work with other versions. For more information about how to install these versions, check the [Pytorch documentation](https://pytorch.org/get-started/previous-versions/#v1121).

The simulation parameters to train and test different scenarios are located in the `simulation_parameters.py` file within the `lib/` directory. The `methods/` directory contains the scripts to train and test all methods: (i) baseline (ii) supervised learning, (iii) unsupervised learning, and (iv) semi-supervised learning. To obtain Fig. 4 of the original paper, semi-supervised learning should be run with different supervised training iterations.

## Additional information
If you decide to use the source code for your research, please make sure to cite our paper:
- J. M. Mateos-Ramos, B. Chatelier, C. Häger, M. F. Keskin, L. L. Magoarou, and H. Wymeersch, "Semi-Supervised End-to-End Learning for Integrated Sensing and Communications," in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN), Stockholm, Sweden, 2024.
