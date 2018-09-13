# -*- coding: UTF-8 -*-
import pickle
from keras.models import load_model
import pandas as pd
from pandas import DataFrame, merge


MODEL_FILENAME = "/Users/dickye/Documents/codes/pycharm/solving_captchas_code_examples/ali_yancheng/yancheng_model.hdf5"
MODEL_LABELS_FILENAME = "/Users/dickye/Documents/codes/pycharm/solving_captchas_code_examples/ali_yancheng/yancheng_model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    mapper = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

predictdata_A = pd.read_table("/info/ali/yancheng/test_A_20171225.txt",index_col=False)
predictdata_B = pd.read_table("/info/ali/yancheng/test_A_20171225.txt",index_col=False)

predictdata=pd.concat([predictdata_A,predictdata_B])

predictdata['tmpKey'] = [1] * len(predictdata.index)

suppInfo = pd.DataFrame({"brand":[x for x in range(1,6)],
                        "tmpKey":[1]*5,
                        }
                       )
test_data=merge(predictdata, suppInfo,on='tmpKey')[['date','day_of_week','brand']]

encoded_X_test=mapper.transform(test_data)
Y_Predict=model.predict(encoded_X_test,batch_size=100,verbose=1)
print Y_Predict
