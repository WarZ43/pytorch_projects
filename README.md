Includes a series of neural net projects to practice conceptual understanding of neural nets and usage of Pytorch UI.

This repository is a collection of deep learning projects exploring both from-scratch implementations and PyTorch-based models. It includes from-scratch(excluding NumPy and math) RNN, LSTM, and numerical Transformer implementations 
with multi-cell / multi-head support and parameterized model size using NumPy matrices for efficient calculations, tested on the classic Airline Passengers dataset for time series prediction (currently limited to single-feature inputs for LSTM and RNN).
Alongside these, a PyTorch equivalent of the LSTM is provided for comparison and validation. The custom implementation matches and sometimes beats the Pytorch equivalent with loss <0.04 compared to the 
Pytorch equivalent's loss <0.05.

The transformer implementation is a numerical transformer with a 2 layer ReLU FFN at the end to handle nonlinearization more effectively. It achieves the lowest MSE of 0.00033 compared to 0.0018 from the LSTM and 0.002 from the RNN. The implementation also includes hyperparameters for multi-head, multi-feature, and batch support.

The repo also contains a Pytorch CNN trained on CIFAR-10, achieving 92.8% test accuracy. Additionally, there is an experimental attempt at creating a DCGAN trained on the CelebA
dataset, aimed at generating realistic face images and exploring generative modeling techniques that is still a work in progress.
