# ------------------------------------------------------------------------------
# define activation functions
relu <- function(x) {
  ifelse(x > 0, x, 0)
}
softplus <- function(x) {
  log(1 + exp(x))
}

swish <- function(x) {
  x * (1 / (1 + exp(-x)))
}

# other options include:
# base::tanh()


# custom activation functions
# relu <- function(x)
#   sapply(x, function(z)
#     max(0, z))
