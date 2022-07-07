# 1D CNN neural network
# 运用Keras一维卷积实现
from keras import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

from models.formation_1d_cnn_force_data import TIME_PERIODS, num_sensors, num_classes, input_shape


class D1_CNN:
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
