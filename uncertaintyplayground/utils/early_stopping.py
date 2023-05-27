import numpy as np

class EarlyStopping:
    """
    Early stopping utility class for model training.

    Stops the training process when a specified performance metric does not improve for a specified number of consecutive epochs.

    Args:
        patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
        compare_fn (callable): Function to compare two values of the validation metric to determine if one is better than the other.
    """

    def __init__(self, patience=10, compare_fn=lambda x, y: x < y):
        self.patience = patience
        self.counter = 0
        self.best_val_metric = np.inf
        self.best_model_state = None
        self.compare_fn = compare_fn

    def __call__(self, val_metric, model):
        if self.compare_fn(val_metric, self.best_val_metric):
            self.best_val_metric = val_metric
            self.counter = 0
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False