#!/usr/bin/env python3

#####################################################################################################
#
#   Author: Carlos Arevalo (caeareva)
#
#   Program execution:
#   python /Users/carevalo/Desktop/nnsp_regression/nnsp_regression.py
#   
#   Description:
#   Data to run program has been provided by by Harrison Kinsley and Daniel Kukieła, authors
#   of Neural Networks from Scratch in Python (NNSP). If data is inputed as a file, an argparse
#   module needs to be included in the program (in this file).
#
#####################################################################################################

### Neral Networks from Scratch in Python (by page 458)
import numpy as np
import pandas as pd
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import layer_dense as ld # Layer_Dense()
import activation_relu as ar # Activation_ReLU()
import activation_linear as al # Activation_Linear()
import loss_meanSquaredError as lmse # Loss_MeanSquaredError
import optimizer_Adam as oa # Optimizer_Adam()

nnfs.init()

# Create dataset
X, y = sine_data()

# Create Dense layer with 1 input feature and 64 output values
dense1 = ld.Layer_Dense(1, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = ar.Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 64 output values
dense2 = ld.Layer_Dense(64, 64)

# Create ReLU activation (to be used with Dense layer):
activation2 = ar.Activation_ReLU()

# Create third Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense3 = ld.Layer_Dense(64, 1)

# Create Linear activation:
activation3 = al.Activation_Linear()

# Create loss function
loss_function = lmse.Loss_MeanSquaredError()

# Create optimizer
optimizer = oa.Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of all the ground truth values
accuracy_precision = np.std(y) / 250

# Create parameters in empty lists
temp_df = pd.DataFrame()
epoch_list = []
acc_list = []
loss_list = []
dataloss_list = []
reg_loss = []
lr_list = []

# Train in loop
for epoch in range(10001):
    
    # Append epochs to list
    epoch_list.append(epoch)

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)

    # Perform a forward pass through third Dense layer
    # takes outputs of activation function of second layer as inputs
    dense3.forward(activation2.output)

    # Perform a forward pass through activation function
    # takes the output of third dense layer here
    activation3.forward(dense3.output)

    # Calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y)
    dataloss_list.append(data_loss)

    # Calculate regularization penalty
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2) + \
        loss_function.regularization_loss(dense3)
    reg_loss.append(regularization_loss)

    # Calculate overall loss
    loss = data_loss + regularization_loss
    loss_list.append(loss)

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) <
                       accuracy_precision)
    acc_list.append(accuracy)
    lr_list.append(optimizer.current_learning_rate)

    if not epoch % 100: 
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' + 
            f'reg_loss: {regularization_loss:.3f}), ' + 
            f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# Add parameters as columns and save dataframe as tsv
temp_df["epoch"] =  epoch_list
temp_df["accuracy"] = acc_list
temp_df["loss"] = loss_list
temp_df["data_loss"] = data_loss
temp_df["regularization_loss"] = reg_loss
temp_df["learning_rate"] = lr_list
temp_df.to_csv(r'/Users/carevalo/Desktop/nnsp_regression/training_parameters.tsv', \
    sep='\t', encoding='utf-8', header=True)

# Plot training parameters
figureHeight=5
figureWidth=4
plt.figure(figsize=(figureWidth, figureHeight)) 

# Define panel dimentions
panelHeight=1
panelWidth=2
relativePanelWidth = panelWidth/figureWidth
relativePanelHeight = panelHeight/figureHeight

# Make figure panels
loss_fig = plt.axes([2/7, 0.7, relativePanelWidth, relativePanelHeight])
acc_fig = plt.axes([2/7, 0.4, relativePanelWidth, relativePanelHeight])
lr_fig = plt.axes([2/7, 0.1, relativePanelWidth, relativePanelHeight])

# Summarize history for Loss
loss_fig.plot(epoch_list, loss_list, \
    linestyle="solid", color="#1770AB")
loss_fig.set_title("Loss")
loss_fig.set_xticks([i for i in range(0, 11000, 2500)])
loss_fig.set_xticklabels([])

# Summarize history for Accuracy
acc_fig.plot(epoch_list, acc_list, \
    linestyle="solid", color="#FF0000")
acc_fig.set_title("Accuracy")
acc_fig.set_xticks([i for i in range(0, 11000, 2500)])
acc_fig.set_xticklabels([])

# Summarize history for Learning rate
lr_fig.plot(epoch_list, lr_list, \
    linestyle="solid", color="#00CD6C")
lr_fig.set_title("Learning rate")
lr_fig.set_xlabel("Epoch")
#lr_fig.set_ylim(0, 0.005)
lr_fig.set_xticks([i for i in range(0, 11000, 2500)])

# Save figures
plt.savefig("/Users/carevalo/Desktop/nnsp_regression/training_parameters.png", dpi=600)

# Plot regression line

# Re-load data
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

# Plot and save figure
figureHeight=4.5
figureWidth=5
plt.figure(figsize=(figureWidth, figureHeight)) 
plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output) 
plt.savefig("/Users/carevalo/Desktop/nnsp_regression/nnsp_regression_line.png", dpi=600)






