import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import ops
import tensorflow

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential()
model.add(layers.Dense(1, activation="relu"))
model.compile(
    optimizer="Adam",
    loss=keras.losses.BinaryCrossentropy()
)
history = model.fit(c, f)

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
