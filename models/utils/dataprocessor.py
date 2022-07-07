from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing


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

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)  # axis = 0:压缩行，对各列求均值，返回 1* n 矩阵
    sigma = np.std(dataset, axis=0)  # axis = 0:计算每一列的标准差
    return (dataset - mu) / sigma


# 显示 confusion matrix
def show_confusion_matrix(validations, predictions, LABELS):
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
    print(dataframe.describe())


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


def plot_acc_loss(history):
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