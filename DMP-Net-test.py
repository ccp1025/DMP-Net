import numpy as np
import pandas as pd
from keras.models import model_from_json


with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')


ceshi = pd.read_csv(r"F:\K\X8.csv", header=None)
ceshi = ceshi.iloc[:, :].values
x_test2 = ceshi[:, 0:1]
X_test3 = x_test2.reshape(9128704, 1, 1)
predicted = loaded_model.predict(X_test3)
Y_pred = pd.DataFrame(predicted)
Y_pred.to_csv(r"F:\K_cnn_result\cnn_Y8.csv", header=False, index=False)

print("------------------------------完成--------------------------------")

