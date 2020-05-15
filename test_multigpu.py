import tensorflow.compat.v1 as tf
from keras.utils.multi_gpu_utils import multi_gpu_model
tf.disable_v2_behavior()
import keras.backend.tensorflow_backend as tfback

  
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1])

predictions = tf.nn.softmax(predictions)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions)

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])


model = multi_gpu_model(model)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])

# probability_model(x_test[:5])