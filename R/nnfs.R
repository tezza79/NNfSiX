library(dplyr)

# set up working dir 
dir <- ""
setwd(dir)

# source nn class
source(file = file.path(getwd(), "nn_classes.R"))

# create x y spiral data
points <- 100
data <- mlbench::mlbench.spirals(points, 1.5, 0.05)
data_xy <-lapply(1:points, function(i) data$x[i,])
plot(data)

# create a dense layer - 2 input features, 3 output values
dense1 <- DenseLayer(2 ,3)
dense1

# Create ReLU activation (to be used with Dense layer):
activation1 <- Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1_output <- forward(obj = dense1, input_data = data_xy)

# fwd pass dense 1 layer output through activation func
dense1_output <- forward(obj = activation1, input_data = dense1_output)

# -----

# create a 2nd dense layer - 2 input features, 3 output values
dense2 <- DenseLayer(3 ,3)
dense2

# Create Softmax activation (to be used with Dense layer):
activation2 <- Activation_Softmax()

# Make a forward pass of our training data through this layer
dense2_output <- forward(obj = dense2, input_data = dense1_output)

# fwd pass dense 1 layer output through activation func
dense2_output <- forward(obj = activation2, input_data = dense2_output)

tail(dense2_output)
