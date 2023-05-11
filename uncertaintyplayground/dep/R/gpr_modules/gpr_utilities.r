# TFDataset & GPR utilies

#' R-squared (R2) Custom Metric
#'
#' This function calculates the R-squared (R2) metric, a measure of how well the model predictions
#' explain the variance in the true target values. The R2 metric ranges from 0 to 1, with higher
#' values indicating better performance. This custom metric can be used during the training of a
#' `keras` model.
#'
#' @param y_true A tensor of the true target values.
#' @param y_pred A tensor of the predicted target values.
#' @return A scalar tensor representing the R-squared (R2) value.
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(tensorflow)
#'
#' # Generate some example data
#' set.seed(123)
#' x <- seq(1, 100)
#' y <- 3 * x + rnorm(100, mean = 0, sd = 10)
#' train_data <- data.frame(x = x, y = y)
#'
#' # Create a simple linear regression model
#' model <- keras_model_sequential() %>%
#'     layer_dense(units = 1, input_shape = 1)
#'
#' # Compile the model with the custom R2 metric
#' model %>% compile(
#'     optimizer = optimizer_adam(lr = 0.01),
#'     loss = "mse",
#'     metrics = list(r2_metric)
#' )
#'
#' # Train the model
#' history <- model %>% fit(
#'     as.matrix(train_data[["x"]]),
#'     as.matrix(train_data[["y"]]),
#'     epochs = 100,
#'     batch_size = 10,
#'     validation_split = 0.2
#' )
#'
#' # Print training history
#' print(history)
#' }
#' @export
r2_metric <- custom_metric("r2", function(y_true, y_pred) {
    unexplained_error <- k_sum((y_true - y_pred)^2)
    total_error <- k_sum((y_true - k_mean(y_true))^2)
    r2 <- 1 - (unexplained_error / total_error)
    return(r2)
})

#' Extract a component from a TensorFlow dataset.
#'
#' This function takes a TensorFlow dataset and a component name ("features",
#' "labels", or "weights") and returns a vector or matrix containing the values
#' of that component for each observation in the dataset.
#'
#' If the component is "features", the resulting matrix will have the same
#' number of rows and columns as the original input to the model.
#'
#' If the component is "weights" and the weights are missing or empty, this
#' function returns NULL.
#'
#' @param tf_data A TensorFlow dataset object.
#' @param component The name of the component to extract ("features", "labels",
#'   or "weights").
#' @return A vector or matrix containing the values of the specified component
#'   for each observation in the dataset, or NULL if the weights component is
#'   missing or empty.
#' @export
#' @examples
#' # Extract the features from a TensorFlow dataset
#' features <- extract_component(tf_data, "features")
#'
#' # Extract the labels from a TensorFlow dataset
#' labels <- extract_component(tf_data, "labels")
#'
#' # Extract the weights from a TensorFlow dataset
#' weights <- extract_component(tf_data, "weights")
extract_component <- function(tf_data, component) {
  # Define a dictionary mapping component names to indices
  component_index <- c("features" = 1, "labels" = 2, "weights" = 3)

  # Check if the component argument is valid
  if (!component %in% names(component_index)) {
    stop("Invalid component name")
  }

  # Get the index of the component
  index <- component_index[[component]]

  # Define a function to extract the component from a batch
  extract_component_from_batch <- function(batch) {
    return(batch[[index]])
  }

  # Iterate through the dataset and extract the component from each batch
  component_list <- reticulate::iterate(tf_data, extract_component_from_batch)

  # Convert the components to R objects
  component_converted <- lapply(component_list, function(x) {
    component_tensor <- x$numpy()

    # If the component is the features, reshape it into the original format
    if (component == "features") {
      n_rows <- dim(component_tensor)[1]
      n_cols <- dim(component_tensor)[2]
      component_tensor <- array(component_tensor, dim = c(n_rows, n_cols))
    }

    return(component_tensor)
  })

  # Concatenate the components into a single vector or matrix
  if (component == "features") {
    component_matrix <- do.call(rbind, component_converted)
    return(component_matrix)
  } else {
    # Check if the weights component exists and has non-zero length
    if (component == "weights" && length(component_converted) == 0) {
      return(NULL)
    } else {
      component_vector <- do.call(c, component_converted)
      return(component_vector)
    }
  }
}
#' Convert a MapDataset object into a list of arrays
#'
#' This function takes a TensorFlow MapDataset object and iterates over its
#' elements, converting each element into an array. The resulting arrays are
#' combined along a new dimension using the abind package.
#'
#' @param dataset A TensorFlow MapDataset object to be converted.
#' @return A list of combined arrays.
convert_map_dataset_to_array <- function(dataset) {
  # Initialize an empty list to store the arrays
  result_list <- list()
  
  # Create a one-shot iterator for the dataset
  iterator <- make_iterator_one_shot(dataset)
  
  # Iterate through the dataset, converting each element into an array
  while (TRUE) {
    element <- tryCatch({
      iterator_get_next(iterator)
    }, error = function(e) NULL)
    
    if (is.null(element)) {
      break
    }
    
    # Convert the element to an array and append it to the result_list
    result_list[[length(result_list) + 1]] <- lapply(element, as.array)
  }
  
  # Combine the arrays along a new dimension
  num_elements <- length(result_list[[1]])
  combined_arrays <- lapply(seq_len(num_elements), function(i) {
    # Concatenate the arrays along the 0th dimension using abind::abind
    do.call(abind::abind, c(lapply(result_list, function(x) x[[i]]), list(along = 0)))
  })
  
  combined_arrays
}
#' Extract a component from a TensorFlow dataset.
#'
#' This function takes a TensorFlow dataset and a component name ("features",
#' "labels", or "weights") and returns a vector or matrix containing the values
#' of that component for each observation in the dataset.
#'
#' If the component is "features", the resulting matrix will have the same
#' number of rows and columns as the original input to the model.
#'
#' If the component is "weights" and the weights are missing or empty, this
#' function returns NULL.
#'
#' @param tf_data A TensorFlow dataset object.
#' @param component The name of the component to extract ("features", "labels",
#'   or "weights").
#' @return A vector or matrix containing the values of the specified component
#'   for each observation in the dataset, or NULL if the weights component is
#'   missing or empty.
#' @export
#' @examples
#' # Extract the features from a TensorFlow dataset
#' features <- extract_component(tf_data, "features")
#'
#' # Extract the labels from a TensorFlow dataset
#' labels <- extract_component(tf_data, "labels")
#'
#' # Extract the weights from a TensorFlow dataset
#' weights <- extract_component(tf_data, "weights")
extract_component <- function(tf_data, component) {
  # Define a dictionary mapping component names to indices
  component_index <- c("features" = 1, "labels" = 2, "weights" = 3)

  # Check if the component argument is valid
  if (!component %in% names(component_index)) {
    stop("Invalid component name")
  }

  # Get the index of the component
  index <- component_index[[component]]

  # Define a function to extract the component from a batch
  extract_component_from_batch <- function(batch) {
    return(batch[[index]])
  }

  # Iterate through the dataset and extract the component from each batch
  component_list <- reticulate::iterate(tf_data, extract_component_from_batch)

  # Convert the components to R objects
  component_converted <- lapply(component_list, function(x) {
    component_tensor <- x$numpy()

    # If the component is the features, reshape it into the original format
    if (component == "features") {
      n_rows <- dim(component_tensor)[1]
      n_cols <- dim(component_tensor)[2]
      component_tensor <- array(component_tensor, dim = c(n_rows, n_cols))
    }

    return(component_tensor)
  })

  # Concatenate the components into a single vector or matrix
  if (component == "features") {
    component_matrix <- do.call(rbind, component_converted)
    return(component_matrix)
  } else {
    # Check if the weights component exists and has non-zero length
    if (component == "weights" && length(component_converted) == 0) {
      return(NULL)
    } else {
      component_vector <- do.call(c, component_converted)
      return(component_vector)
    }
  }
}