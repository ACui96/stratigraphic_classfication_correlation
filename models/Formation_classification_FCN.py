import pickle

import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
import pandas as pd
from sklearn import preprocessing
from keras.layers import Dense, Conv1D, Flatten

from classifiers import fcn
from models.utils.dataprocessor import feature_normalize, create_segments_and_labels, plot_acc_loss

TIME_PERIODS = 160
STEP_DISTANCE = 2
features = ['Depth', 'GR', 'INPEFA', 'AC', 'SP', 'GR_MED_3', 'GR_MED_5', 'GR_MED_7',
            'GR_MED_9', 'GR_MED_11', 'GR_MED_13', 'GR_MED_15', 'GR_MED_17',
            'GR_MED_19']
fea_num = len(features)

# 载入数据
def read_data(path, target):
    df = pd.read_csv(path)
    print(df.head())
    classes = len(df[target].unique())
    # features = df.columns.drop([target, 'Well Name'])
    for f in features:
        df = df.round({f: 3})  # 保留3位小数点
        df[f] = feature_normalize(df[f])

    LABEL = target + "_encoded"
    le = preprocessing.LabelEncoder()
    # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
    df[LABEL] = le.fit_transform(df[target].values.ravel())

    df_train = df[df['Well Name'] > 2]
    df_test = df[df['Well Name'] == 0]
    df_validation = df[df['Well Name'] <= 2]

    x_train, y_train = create_segments_and_labels(df_train,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL,
                                                  features)

    x_valid, y_valid = create_segments_and_labels(df_validation,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL,
                                                  features)

    x_test, y_test = create_segments_and_labels(df_test,
                                                TIME_PERIODS,
                                                STEP_DISTANCE,
                                                LABEL,
                                                features)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, classes





# train_x, train_y, valid_x, valid_y, test_x, test_y, classes = read_data(r'D:\workspace\scientificProject\stratigraphic_classfication_correlation\data\force-train.csv')
train_x, train_y, valid_x, valid_y, test_x, test_y, classes = read_data('../data/ciflog_scalared_fetures_med9.csv',
                                                                        target='Formation')
output_directory = '/out'
input_shape = (TIME_PERIODS, fea_num)
nb_classes = classes
verbose = True
classifier = fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
y_true = np.argmax(test_y, axis=1)
classifier.fit(train_x, train_y, test_x, test_y, y_true)

