# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np

# 读取训练数据
data = pd.read_csv("testTrain/train.csv")
# 显示数据属性：有多少行多少列以及每列的数据类型等
# data.info()

# sex字段如果时male赋值为1，否则赋值为0
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# 缺失字段填充为0
data = data.fillna(0)

# 增加一列'身亡'
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))

# 提取部分特征字段，数据集X为训练数据集，数据集Y为验证数据集
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()

# 随机取20%的数据样本
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

with tf.name_scope('input'):
    # 定义占位符，因为算子定义时不直接参数运算（Feed机制），所以输入就用占位符表示
    X = tf.placeholder(tf.float32, shape=[None, 6]) # None代表可以输入任意条6元数据
Y = tf.placeholder(tf.float32, shape=[None, 2])

with tf.name_scope('classifier'):
    # 声明变量
    W = tf.Variable(tf.random_normal([6, 2]), name='weights')
    b = tf.Variable(tf.zeros([2], name='bias'))
    # 构造前向传播计
    y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
    # 添加直方图参数概要记录算子
    tf.summary.histogram('weights', W);
    tf.summary.histogram('bias', b);

with tf.name_scope('cost'):
    # 声明代价函数
    cross_entropy = - tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1) # 根据交叉熵公式进行计算
    cost = tf.reduce_mean(cross_entropy) # 取所有样本的交叉熵平均值作为批量样本代价
    # 添加损失代价标量概要
    tf.summary.scalar('loss', cost)

# 使用随即梯度下降算法优化器来最小化代价
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', acc_op)

# 构建训练迭代过程
with tf.Session() as sess:
    # 初始化所有变量
    tf.global_variables_initializer().run()

    # 创建概要写入操作
    # TensorBoard可以通过命令'Tensorboard --logdir=./logs'来启动
    writer = tf.summary.FileWriter('./logs', sess.graph)
    # 方便起见，合并所有概要算子
    merged = tf.summary.merge_all()

    # 使用Saver保存训练好的模型，不必每次都需要训练
    saver = tf.train.Saver()

    # 进行训练（迭代10轮）
    for epoch in range(10) :
        total_loss = 0
        for i in range(len(X_train)) :
            feed = {X: [X_train[i]], Y: [Y_train[i]]}
            # 执行运算
            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
            summary, accuracy = sess.run([merged, acc_op],
                                     feed_dict={X: [X_train[i]], Y: [Y_train[i]]})
        writer.add_summary(summary, epoch)
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print 'Training complete!'

    # 评估校验数据集上的准确性
    pred = sess.run(y_pred, feed_dict={X:X_test})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set: %.9f" % accuracy)
    savePath = saver.save(sess, "model.ckpt")


testdata = pd.read_csv('testTrain/test.csv')
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s : 1 if s == 'male' else 0)

X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

with tf.Session() as sess2:
    # 加载存档
    saver.restore(sess2, 'model.ckpt')
    print W.value(), b.value()

    # 正向传播计算
    predictions = np.argmax(sess2.run(y_pred, feed_dict={X: X_test}), 1)

# 构建提交结果的数据结构，并将结果存储为csv文件
submission = pd.DataFrame({
    "PassengerId" : testdata["PassengerId"],
    "Survived" : predictions
})

submission.to_csv("titanic-submission.csv", index=False)