import tensorflow.keras as K
import tensorflow.keras.layers as KL


class MNIST(K.Model):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__()
        self.config = config
        self.conv1 = KL.Conv2D(32, 3, activation="relu")
        self.flatten = KL.Flatten()
        self.d1 = KL.Dense(128, activation="relu")
        self.d2 = KL.Dense(10, activation="softmax")

    def call(self, x: K.Tensor) -> KL.Layer:
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
