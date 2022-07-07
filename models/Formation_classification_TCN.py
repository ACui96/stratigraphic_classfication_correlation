import pickle

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
import pandas as pd
from sklearn import preprocessing
from keras.layers import Dense, Conv1D, Flatten

from models.utils.dataprocessor import feature_normalize, create_segments_and_labels, plot_acc_loss

TIME_PERIODS = 160
STEP_DISTANCE = 2
# features_ = ['DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC',
#             'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC',
#             'SP', 'BS', 'ROP', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'FORMATION_encoded',
#             'WELL_encoded', 'LITHOLOGY']
# to_normalize_features_ = [
#     'DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC',
#     'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC',
#     'SP', 'BS', 'ROP', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC'
# ]
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


# 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    return o





# 序列模型
def TCN(train_x, train_y, valid_x, valid_y, test_x, test_y, classes):
    # TIME_PERIODS = 160
    # STEP_DISTANCE = 80

    inputs = Input(shape=(TIME_PERIODS, fea_num))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)
    ]

    # 训练模型
    history = model.fit(train_x,
                        train_y,
                        batch_size=500,
                        epochs=100,
                        verbose=2,
                        validation_data=(valid_x, valid_y),
                        callbacks=callbacks_list)


    with open('../data/history.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # acc - loss curve
    plot_acc_loss(history)
    # 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])



# train_x, train_y, valid_x, valid_y, test_x, test_y, classes = read_data(r'D:\workspace\scientificProject\stratigraphic_classfication_correlation\data\force-train.csv')
train_x, train_y, valid_x, valid_y, test_x, test_y, classes = read_data('../data/ciflog_scalared_fetures_med9.csv',
                                                                        target='Formation')
TCN(train_x, train_y, valid_x, valid_y, test_x, test_y, classes)
