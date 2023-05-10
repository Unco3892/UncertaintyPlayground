from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

class KernelLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(KernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(KernelLayer, self).build(input_shape)

    def call(self, x):
        # Define the kernel function as a squared exponential
        def kernel(x1, x2):
            sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
            return np.exp(-0.5 * sqdist)

        # Compute the kernel matrix
        kernel_matrix = kernel(x, x)

        # Multiply the kernel matrix by the kernel weights
        output = K.dot(K.constant(kernel_matrix), self.kernel)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# Generate some sample data
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000)

# Define the model architecture
input_layer = Input(shape=(10,))
hidden_layer = KernelLayer(output_dim=10)(input_layer)
output_layer = Dense(units=1)(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
