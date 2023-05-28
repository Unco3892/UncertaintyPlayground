import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os

class BaseTrainer:
    """
    Base trainer class for model training.

    Args:
        X (np.ndarray or torch.Tensor): Input data of shape (num_samples, num_features).
        y (np.ndarray or torch.Tensor): Target data of shape (num_samples,).
        sample_weights (np.ndarray or torch.Tensor): Optional sample weights of shape (num_samples,).
        test_size (float): The proportion of the data to include in the validation set.
        random_state (int): The seed used by the random number generator.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer_fn_name (str): Name of the optimizer function from the `torch.optim` module.
        lr (float): Learning rate for the optimizer.
        use_scheduler (bool): Whether to use a learning rate scheduler (`not yet fully supported`).
        patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
        dtype (torch.dtype): Data type to use for the tensors.

    Attributes:
        X (torch.Tensor): Input data tensor.
        y (torch.Tensor): Target data tensor.
        sample_weights (torch.Tensor): Sample weights tensor.
        test_size (float): Proportion of data to include in the validation set.
        random_state (int): Seed used by the random number generator.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer_fn_name (str): Name of the optimizer function.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
        dtype (torch.dtype): Data type of the tensors.
        device (torch.device): Device (GPU if available, otherwise CPU).
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    """

    def __init__(
            self,
            X,
            y,
            sample_weights=None,
            test_size=0.2,
            random_state=42,
            num_epochs=50,
            batch_size=256,
            optimizer_fn_name="Adam",
            lr=0.01,
            use_scheduler=False,
            patience=10,
            dtype=torch.float32
    ):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.test_size = test_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_fn_name = optimizer_fn_name
        self.lr = lr
        self.patience = patience
        self.dtype = dtype
        self.use_scheduler = use_scheduler

        # Setting for early stopping
        self.best_epoch = -1
        self.best_val_mse = np.inf

        # Choose device (GPU if available, otherwise CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Convert input tensors to the correct type
        self.prepare_inputs()

        # Split data into training and validation sets
        self.split_data()

        # Create DataLoader for training data
        self.prepare_dataloader()

    def prepare_inputs(self):
        """
        Convert input data to the correct type and format.
        """
        # Convert X to a tensor if it's a numpy array
        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X).to(self.dtype)

        # Convert y to a tensor if it's a list or numpy array
        if isinstance(self.y, (list, np.ndarray)):
            self.y = torch.tensor(self.y, dtype=self.dtype)

        # Check if sample_weights is a tensor, numpy array, or list
        if self.sample_weights is not None:
            if isinstance(self.sample_weights, (np.ndarray, list)):
                self.sample_weights = torch.tensor(
                    self.sample_weights, dtype=self.dtype)

    def split_data(self, test_size=0.2):
        """
        Split the data into training and validation sets.

        Args:
            test_size (float): Proportion of data to include in the validation set.
        """
        if self.sample_weights is None:
            self.sample_weights = torch.ones(self.X.shape[0], dtype=self.dtype)

        self.X_train, self.X_val, self.y_train, self.y_val, self.sample_weights_train, self.sample_weights_val = \
            train_test_split(self.X, self.y, self.sample_weights,
                             test_size=test_size, random_state=self.random_state)

    def custom_lr_scheduler(self, epoch):
        """
        Custom learning rate scheduler function.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate for the epoch.
        """
        if epoch < 3:
            return 1 - 0.1 * epoch / self.lr
        else:
            return 0.2 / self.lr

    def prepare_dataloader(self):
        """
        Prepare the DataLoader for training data.
        """
        # Use all available CPU cores or default to 1 if not detected
        num_workers = os.cpu_count() - 1 or 1
        train_dataset = TensorDataset(
            self.X_train, self.y_train, self.sample_weights_train)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
