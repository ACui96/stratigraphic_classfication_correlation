from __future__ import print_function

from keras import Input
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, add, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


# 规范化数据：使各个数据处于同一量级
# 中心化（又叫零均值化）：是指变量减去它的均值。其实就是一个平移的过程，平移后所有数据的中心是（0，0）
# 标准化（又叫归一化）： 是指数值减去均值，再除以标准差。
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)  # axis = 0:压缩行，对各列求均值，返回 1* n 矩阵
    sigma = np.std(dataset, axis=0)  # axis = 0:计算每一列的标准差
    return (dataset - mu) / sigma


# 显示 confusion matrix
def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(16, 14))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


# 打印传入的原始数据  显示的行数由preview_rows确定
def show_basic_dataframe_info(dataframe,
                              preview_rows=20):
    """
    This function shows basic information for the given dataframe
    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview
    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())


# 转化为浮点数
def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


# 图表显示
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(True)  # 设置x轴坐标轴不可见
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])  # 限制显示的范围
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


# 图表显示
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,  # 将一个figure分成3个子图，分别显示
                                        figsize=(15, 10),
                                        sharex=True)
    plot_axis(ax0, data['timestamp'], data['co-fli'], 'CO-Filtering')  # 自定义函数
    plot_axis(ax1, data['timestamp'], data['smog-fli'], 'Smog-Filtering')
    plot_axis(ax2, data['timestamp'], data['t-fli'], 'Temperature-Filtering')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    # subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # 其中left、bottom、right、top围成的区域就是子图的区域。wspace、hspace分别表示子图之间左右、上下的间距。


# 接收read_data()处理好的txt文件里面的数据，转化为（reshape）cnn能够识别的数据帧
def create_segments_and_labels(df, time_steps, step, label_name, features: list):
    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
        features: 训练特征
    Returns:
        reshaped_segments
        labels:
    """

    # 传入特征的个数
    N_FEATURES = len(features)
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        segment = []
        for feature in features:
            segment.append(df[feature].values[i:i + time_steps])
        # 寻找一个 segment 中出现最多的标签
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append(segment)
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

#残差块
def ResBlock(x,filters,kernel_size,dilation_rate):
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=add([r,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o

# 序列模型 TCN
def TCN(train_x, train_y, valid_x, valid_y, test_x, test_y):
    inputs = Input(shape=(28, 28))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input=inputs, output=x)
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(train_x, train_y, batch_size=500, nb_epoch=30, verbose=2, validation_data=(valid_x, valid_y))
    # 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])


def FCN(train_x, train_y, valid_x, valid_y, test_x, test_y):
    nb_classes = num_classes
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)
    output_directory = '../h5'
    file_path = output_directory + 'best_model.hdf5'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)

    callbacks = [reduce_lr, model_checkpoint]


# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# The number of steps within one time segment
# one time segment的长度,因为只有三个传感器的值，所以宽度固定为3
TIME_PERIODS = 160
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
# time segment 每次移动的步长
STEP_DISTANCE = 80

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
# 读取原始数据
df = pd.read_csv(r'D:\workspace\pycharm\welllogAI\data\处理后的FORCE数据集\force-train.csv')
LABELS = df['GROUP_encoded'].unique()
# Describe the data
# 图表显示
# 原始数据 图表/数值 显示开关 dis_switch

dis_switch = True

if dis_switch:
    show_basic_dataframe_info(df, 20)

    # 柱状图显示

    df['LITHOLOGY'].value_counts().plot(kind='bar',
                                        title='Training Examples by Formation Type')

    plt.show()

    df['WELL_encoded'].value_counts().plot(kind='bar',
                                           title='Training Examples by Well Name')
    plt.show()

# Define column name of the label vector
LABEL = "GroupEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
# Add a new column to the existing DataFrame with the encoded values
# 将数据中的字符标签转化为数字标签
df[LABEL] = le.fit_transform(df["GROUP_encoded"].values.ravel())

# %%

print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
# 区分训练集和测试集
# 标号为1的为训练集，标号为2的为测试集
df_test = df[df['WELL_encoded'] <= 7]
df_train = df[df['WELL_encoded'] > 7]

# already normalized

# Round in order to comply to NSNumber from iOS

# features = ['Depth', 'GR', 'AC', 'SP', 'INPEFA' ]
features = ['DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC',
            'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC',
            'SP', 'BS', 'ROP', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'FORMATION_encoded',
            'WELL_encoded', 'LITHOLOGY']
for f in features:
    df_train = df_train.round({f: 3})  # 保留3位小数点
    # Normalize features for training data set
    df_train[f] = feature_normalize(df[f])

# Reshape the training data into segments
# so that they can be processed by the network
# x_train的数据为80×4的二维矩阵，y_train的数据为x_train数据对应的标签
# features = ['Depth', 'GR', 'AC', 'SP', 'INPEFA', 'GR_MED_3', 'GR_MED_5', 'GR_MED_7', 'GR_MED_9', 'GR_MED_11',
#             'GR_MED_13', 'GR_MED_15', 'GR_MED_17', 'GR_MED_19', ]

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL,
                                              features)

# %%


print("\n--- Reshape data to be accepted by Keras ---\n")

print('x_train shape: ', x_train.shape)

print(x_train.shape[0], 'training samples')

# Inspect y dataq
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
# num_time_periods:TIME_PERIODS x_train的行数  num_sensors:传感器个数 x_train列数
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
# 打印数据中所有的标签
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
# keras不支持多维矩阵的输入，所以要将80×3的二维矩阵转化为长度为240的一维矩阵
input_shape = (num_time_periods * num_sensors)  # 80×3
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

# Convert type for Keras otherwise Keras cannot process the data
# 转化为keras能够识别的float32类型
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding：待定概念
# One-hot encoding of y_train labels (only execute once!只需执行一次)
# 待定，可能是x_train转换完成之后，y_train也要完成相应的转换
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)

# %%


print("\n--- Create neural network model ---\n")

# 1D CNN neural network
# 运用Keras一维卷积实现
model_m = Sequential()
# 输入数据（待定）
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# 第一次卷积层 输入矩阵大小：80×3 输出矩阵大小：71×100
# kernel/patch size:10 filter size:100
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# 第二次卷积层  输入矩阵大小：71×100 输出矩阵大小：62×100
# kernel/patch size:10 filter size:100
model_m.add(Conv1D(100, 10, activation='relu'))
# 最大值池化层 输入矩阵大小：62×100 stride size（步长）：3
# 输出矩阵：20×100
model_m.add(MaxPooling1D(3))
# 第三次卷积 输入矩阵大小：20×160 输出11×160
# kernel/patch size:10 filter size:160 
model_m.add(Conv1D(160, 10, activation='relu'))
# 第四次卷积 输入矩阵大小：11×160 输出2×160
model_m.add(Conv1D(160, 10, activation='relu'))
# 平均值池化层  输出1×160
model_m.add(GlobalAveragePooling1D())
# Dropout层
# 为减少过度拟合，部分数据被随机置0,在这里设置的为0.5（50%的数据被随机置0）
model_m.add(Dropout(0.5))
# fully connected layer
# 使用softmax的激励函数
# 输入1×160 输出1×6
model_m.add(Dense(num_classes, activation='softmax'))

print(model_m.summary())
# Display:
# Accuracy on training data
# Accuracy on test data

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
# 存放神经网络训练模型的文件（即后缀名为.h5的文件，放在该目录下的h5文件夹），
# 生成好的模型其他文件可以直接调用，而不用再次进行用神经网络进行训练
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='h5/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
# 每个batch的大小，也就是说每一个batch里面有400个传感器检测数据
BATCH_SIZE = 512
# 训练次数,全部数据经过神经网络的遍数
EPOCHS = 100

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
# model_m.fit():调用该函数，神经网络开始训练，前面只是将所有参数都设置好。
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      # 在训练集的数据中将数据以8：2的比例分开（2/10=0.2），
                      # 8的进行训练（train），2的进行核对（validation ）
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
# 训练集数据相关参数显示
plt.figure(figsize=(6, 4))

# plt.plot(history.history['acc'], "g--", label="Accuracy of training data")         # 版本不同传入不同的参数
# plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")

plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# %%

print("\n--- Check against test data ---\n")

# Normalize features for training data set
# 神经网络已经训练好，接下来是拿测试集进行测试。
# 测试集的数据处理，类似训练集

for f in features:
    df_test = df_test.round({f: 3})
    df_train[f] = feature_normalize(df[f])

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL,
                                            features)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

# 调用评估函数测试测试集数据的准确度
score = model_m.evaluate(x_test, y_test, verbose=1)

# 测试集的参数（accuracy，loss）的显示
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)
# 训练集的confusion matrix的显示
show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))
