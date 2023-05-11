# design the multi-modal approach in `R`
# let's apply this for a single modality which is tabular data
# afterwards, I can read the images and the tabular data and cast them into the
# same object

# load all the necessary libraries
library(tidyverse)

# read the data
train_data <-
  read_csv(here::here("data/metadata/train_data.csv"))

sample_data <- head(train_data, 100)

# we set the training parameters
epochs <- 1
batch_size <- 5
sigma_squared_primes <- nrow(sample_data) / batch_size

# Initialize all the necessary variables
# We assume that the network has the following number of nodes `c(4,10,20,1)`
# We have to generate the weights and biases, to do so, First, define the
# network architecture
# Number of nodes in the layers of the encoder
encoder_nodes <- c(4, 10, 20)
# Number of nodes in the layers of the decoder
decoder_nodes <- c(20, 10, 1)

# Calculate the total number of weights and biases for the encoder
num_weights_and_biases_encoder <- sum((encoder_nodes[1:length(encoder_nodes) - 1] + 1) * encoder_nodes[2:length(encoder_nodes)])

# Calculate the total number of weights and biases for the decoder
num_weights_and_biases_decoder <- sum((decoder_nodes[1:length(decoder_nodes) - 1] + 1) * decoder_nodes[2:length(decoder_nodes)])

# Set the seed for reproducibility
set.seed(1)

# Generate the weights and biases for the encoder
theta <- rnorm(num_weights_and_biases_encoder)

# Generate the weights and biases for the decoder
phi <- rnorm(num_weights_and_biases_decoder)

# z_theta <-
#

# build a neural network from scratch
