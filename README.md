#Pytorch Projects
Includes a series of neural net projects to practice conceptual understanding of neural nets and usage of Pytorch UI.

This repository is a collection of deep learning projects exploring both from-scratch implementations and PyTorch-based models. It includes from-scratch(excluding NumPy and math) RNN and LSTM implementations 
with multi-cell support using NumPy matrices for efficient calculations, tested on the classic Airline Passengers dataset for time series prediction (currently limited to single-feature inputs).
Alongside these, a PyTorch equivalent of the LSTM is provided for comparison and validation. The custom implementation matches and sometimes beats the Pytorch equivalent with loss <0.04 compared to the 
Pytorch equivalent's loss <0.05. The repo also contains a Pytorch CNN trained on CIFAR-10, achieving 92.8% test accuracy. Additionally, there is an experimental attempt at creating a DCGAN trained on the CelebA
dataset, aimed at generating realistic face images and exploring generative modeling techniques that is still a work in progress.
