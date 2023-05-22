# Distributed Machine Learning

This project involves the distributed training of a machine learning model using PyTorch.

## Installation

Before running the code, make sure to install the following software and libraries:

1. [Anaconda](https://www.anaconda.com/products/distribution)

2. Python version 3.10.11 (recommended). Python can be installed through Anaconda.
   ```
   conda create --name DC_Project_AI python=3.10.11
   conda activate DC_Project_AI
   ```
3. PyTorch. After installing Python and Anaconda, open the terminal and type the following command to install PyTorch:

   ```
   pip install torch
   ```

4. Torchvision. To install torchvision, use the following command in the terminal:
   ```
   pip install torchvision
   ```

## Running the Code

After installing the necessary software, you can run the `ftmultigpu.py` file with the following command:

```
python ftmultigpu.py 10 2
```

Here, `arg1` is the number of epochs you want to train the model for, and `arg2` is after how many epochs you want to save the snapshot of the model.

## Project Architecture

This project employs the distributed training of a machine learning model. PyTorch is used as the primary framework. The model we used consist of three layers, the input layers with 784 neurons, two middle layers with 512 neurons, and an output layer with 10 neurons. The dataset that we trained the model on was FashionMNIST: this includes 70,000 images of 28x28 dimension.
