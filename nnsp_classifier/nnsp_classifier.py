#!/usr/bin/env python3

#####################################################################################################
#
#   Author: Carlos Arevalo (caeareva)
#
#   Program execution:
#   python /Users/carevalo/Desktop/nnsp_classifier/nnsp_classifier.py
#   
#   Description:
#   Data to run program has been provided by by Harrison Kinsley and Daniel Kukieła, authors
#   of Neural Networks from Scratch in Python (NNSP). If data is inputed as a file, an argparse
#   module needs to be included in the program (in this file).
#
#####################################################################################################

### Neral Networks from Scratch in Python (by page 407)

import numpy as np
import pandas as pd
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import layer_dense as ld # Layer_Dense()
import activation_relu as ar # Activation_ReLU()
import activation_sigmoid as asig # Activation_Sigmoid # import loss_common as lc
import loss_binary_crossEntropy as lbce # Loss_BinaryCrossentropy()
import optimizer_Adam as oa # Optimizer_Adam()

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

# Create Dense layer with 2 input features and 64 output values 
dense1 = ld.Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):

activation1 = ar.Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = ld.Layer_Dense(64, 1)

# Create Sigmoid activation:
activation2 = asig.Activation_Sigmoid()

# Create loss function
loss_function = lbce.Loss_BinaryCrossentropy()

# Create optimizer
optimizer = oa.Optimizer_Adam(decay=5e-7) 


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
    
    # Calculate the data loss
    data_loss = loss_function.calculate(activation2.output, y)
    dataloss_list.append(data_loss)

    # Calculate regularization penalty
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)
    reg_loss.append(regularization_loss)
    
    # Calculate overall loss
    loss = data_loss + regularization_loss
    loss_list.append(loss)

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    acc_list.append(accuracy)
    lr_list.append(optimizer.current_learning_rate)
    
    # Print parameters
    if not epoch % 100:    
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Add parameters as columns and save dataframe as tsv
temp_df["epoch"] =  epoch_list
temp_df["accuracy"] = acc_list
temp_df["loss"] = loss_list
temp_df["data_loss"] = data_loss
temp_df["regularization_loss"] = reg_loss
temp_df["learning_rate"] = lr_list
temp_df.to_csv(r'/Users/carevalo/Desktop/nnsp_classifier/training_parameters.tsv', \
    sep='\t', encoding='utf-8', header=True)

# Visualize training parameters above

# Define figure dimentions
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
#lr_temp = [f"{n:.3f}" for n in lr_list] # 1 sig figure
lr_fig.plot(epoch_list, lr_list, \
    linestyle="solid", color="#00CD6C")
lr_fig.set_title("Learning rate")
lr_fig.set_xlabel("Epoch")
#lr_fig.set_ylim(0, max(lr_temp))
lr_fig.set_xticks([i for i in range(0, 11000, 2500)])

# Save figure
plt.savefig("/Users/carevalo/Desktop/nnsp_classifier/training_parameters.png", dpi=600)

# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5)*1
accuracy = np.mean(predictions == y_test)

# Print validation parameters
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
