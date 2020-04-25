# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# nn classes & methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Dense Layer Classes & Methods
DenseLayer <- setClass(
  # Set the name of the Class
  "DenseLayer", 
  
  # Define the slots
  representation(
    inputs_n = 'numeric', 
    neurons = 'numeric',
    biases = 'numeric',
    weights = 'list'
  )
)

## define the constructor
## Note that we don't give an argument to the ctor 
## to init the private field
setMethod (f = "initialize", 
           signature  = "DenseLayer",
           definition = function (.Object,
                                  inputs_n, neurons) {
             .Object@neurons <- neurons
             .Object@inputs_n <-inputs_n
             .Object@biases <-rep(0, neurons)
             .Object@weights <- lapply(1:neurons, function(x){
               runif(inputs_n, min = -1, max = 1)  * 0.01    
             })
             return (.Object)
           })

# forward pass method
setGeneric("forward", function(obj, input_data) standardGeneric("forward"))
setMethod("forward", "DenseLayer", function(obj, input_data) {
  calc <-  do.call(rbind, input_data) %*% t(do.call(rbind, obj@weights)) 
  calc <- t(calc) + obj@biases
  calc <- lapply(seq_len(ncol(calc)), function(i) calc[,i])
  return(calc)
})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Activation ReLU Classes & Methods
Activation_ReLU <- setClass(
  # Set the name of the Class
  "Activation_ReLU", 
  
  # Define the slots
  representation(
    inputs = 'list'
  )
)


setMethod("forward", "Activation_ReLU", function(obj, input_data) {
  lapply(input_data, function(x){
    inp <- x
    x[inp < 0] <- 0
    return(x)  
  })
})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Activation Softmax Classes & Methods
Activation_Softmax <- setClass(
  # Set the name of the Class
  "Activation_Softmax", 
  
  # Define the slots
  representation(
    inputs = 'list'
  )
)

setMethod("forward", "Activation_Softmax", function(obj, input_data) {
  exp_vals <- lapply(input_data, function(x) exp(x - max(x)))
  norm_base <- lapply(exp_vals, sum)
  norm_exp <- mapply(function(x, y) x/y, x = exp_vals, y =norm_base)
  norm_exp <- lapply(seq_len(ncol(norm_exp)), function(i) norm_exp[,i])
  return(norm_exp)
  })
