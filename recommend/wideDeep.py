import tensorflow as tf


# 构建wide&deep 模型
class WideAndDeepModel:
    def __init__(self, wide_length, deep_length, deep_last_layer_Len, softmax_label):
        # 一 首先确定 输入部分，包括 1 wide部分，2 deep部分以及 3标签信息y
        self.input_wide_part = tf.palaceholder(tf.float32, shape=[None, wide_length], name='input_wide_part')

        self.input_deep_part = tf.placeholder(tf.float32, shape=[None, deep_length], name='input_deep_part')

        self.input_y = tf.placeholder(tf.float32, shape=[None, softmax_label], name='input_y')

        # 二 定义deep部分的网络结构 #TODO 服务部署脚本
        with tf.name_scope('deep_part'):
            w_x1 = tf.Variable(tf.random_normal([wide_length, 256], stddev=0.03), name='w_x1')
            b_x1 = tf.Variable(tf.random_normal([256]), name='b_x1')

            w_x2 = tf.Variable(tf.random_normal([256, deep_last_layer_Len], stddev=0.03), name='w_x2')
            b_x2 = tf.Variable(tf.random_normal([deep_last_layer_Len]), name='b_x2')

            z1 = tf.add(tf.matmul(self.input_wide_part, w_x1), b_x1)
            a1 = tf.nn.relu(z1)
            self.deep__logits = tf.add(tf.matmul(a1, w_x2), b_x2)

        # 三 定义wide部分的网络结构
        with tf.name_scope('wide_part'):
            weights = tf.Variable(tf.truncated_normal([deep_last_layer_Len + wide_length, softmax_label]))
            biases = tf.Variable(tf.zeros([softmax_label]))
            self.wide_and_deep = tf.concat([self.deep__logits, self.input_wide_part], axis=1)
            self.wide_and_deep_logits = tf.add(tf.matmul(self.wide_and_deep, weights), biases)
            self.predictions = tf.argmax(self.wide_and_deep_logits, 1, name="prediction")

        # 四 定义损失函数
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.wide_and_deep_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        # 定义准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


'''
训练部分代码
'''
import pandas as pd
import numpy as np
import csv


# 1 读取训练数据和标签
def load_data_and_labels(path):
    data = []
    y = []
    total_q = []
    with open(path, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='""')
        for row in rdr:
            emb_val = row[4].split(';')
            emv_val_f = [float(i) for i in emb_val]
            cate_emvb = row[5].split(';')
            cate_emb_val_f = [float(i) for i in cate_emvb]

            total_q.append((int(row[3])))
            data.append(emv_val_f + cate_emb_val_f)
            y.append(float(row[3]))
    data = np.array(data)
    total_q = np.asarray(total_q)
    y.append(float(row[1]))
    bins = pd.qcut(y, 50, retbins=True)

# 2 将标签转为数值区间
def convert_label44er5t
# 3 将标签转为one-hot encoding
# 4 根据训练数据大小生成batch
# 5 模型训练数据路径
# 6 设定模型训练参数
# 7 定义辅助参数
# 8 读取训练数据
# 9 输出文件和临时checkpoint
# 10 初始化所有变量
# 11 为每一次的新训练都生成batch,size
# 12 保存check-point
