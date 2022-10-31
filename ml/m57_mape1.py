import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import tensorflow as tf
import keras


y_true = np.array([1.,2.,3.])
y_pred = np.array([2.,4.,2.])

# ==================== mae ========================= #
mae = mean_absolute_error(
    y_true, y_pred
)
print('mae : ', mae)    # 1.3333333333333333


# ==================== mape ========================= #
mape = mean_absolute_percentage_error(
    y_true, y_pred
)
print('mape : ', mape) # 0.7777777777777778

# ==================== keras_mape ========================= #
mape_tf = keras.metrics.mean_absolute_percentage_error(
    y_true, y_pred
)
# print('mape_tf : ', mape_tf)    # tf.Tensor(77.77777777777779, shape=(), dtype=float64)
print('mape_tf : ', mape_tf.numpy())    # mape_tf :  77.77777777777779


