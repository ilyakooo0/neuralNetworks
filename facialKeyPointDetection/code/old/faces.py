import tensorflow as tf
import csv
from random import sample


dim = 96

cor = tf.placeholder(tf.float32, [None, 30])
isZeroMuls = tf.placeholder(tf.float32, [None, 30])

x = tf.placeholder(tf.float32, [None, dim*dim])

xInp = tf.reshape(x, [-1, dim, dim, 1])

filter1 = tf.Variable(tf.random_normal([7,7,1,16], stddev=0.1), name="filter1")
biases1 = tf.Variable(tf.random_normal([16], stddev=0.1), name="biases1")
conv1 = tf.nn.relu(tf.nn.conv2d(xInp, filter1, [1,1,1,1], padding="VALID") + biases1)
pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], "SAME") # 45

filter2 = tf.Variable(tf.random_normal([6,6,16,32], stddev=0.1), name="filter2")
biases2 = tf.Variable(tf.random_normal([32], stddev=0.1), name="biases2")
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2, [1,1,1,1], "VALID") + biases2)
pool2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], "SAME") # 20

filter3 = tf.Variable(tf.random_normal([5,5,32, 64]), name="filter3")
biases3 = tf.Variable(tf.random_normal([64], stddev=0.1), name="biases3")
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, filter3, [1,1,1,1], "VALID") + biases3)
pool3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], "SAME") # 8

filter4 = tf.Variable(tf.random_normal([5,5,64, 256]), name="filter4")
biases4 = tf.Variable(tf.random_normal([256], stddev=0.1), name="biases4")
conv4 = tf.nn.relu(tf.nn.conv2d(pool3, filter4, [1,1,1,1], "VALID") + biases4)
pool4 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], "SAME") # 2

filter5 = tf.Variable(tf.random_normal([1, 1, 256, 512]), name="filter5")
biases5 = tf.Variable(tf.random_normal([512], stddev=0.1), name="biases5")
conv5 = tf.nn.relu(tf.nn.conv2d(pool4, filter5, [1,2,2,1], "VALID") + biases5) # 1

outFromConv = tf.reshape(conv5, [-1, 512])

keepRate = tf.placeholder(tf.float32)

fcw1 = tf.Variable(tf.random_normal([512, 2048], stddev=0.1), name="fullyConnectedWeights1")
fcb1 = tf.Variable(tf.random_normal([2048], stddev=0.1), name="fullyConnectedBiases1")
fc1 = tf.nn.dropout(tf.matmul(outFromConv, fcw1) + fcb1, keepRate)

fcw2 = tf.Variable(tf.random_normal([2048, 512], stddev=0.1), name="fullyConnectedWeights2")
fcb2 = tf.Variable(tf.random_normal([512], stddev=0.1), name="fullyConnectedBiases2")
fc2 = tf.nn.dropout(tf.matmul(fc1, fcw2) + fcb2, keepRate)

fcw3 = tf.Variable(tf.random_normal([512, 256], stddev=0.1), name="fullyConnectedWeights3")
fcb3 = tf.Variable(tf.random_normal([256], stddev=0.1), name="fullyConnectedBiases3")
fc3 = tf.nn.dropout(tf.matmul(fc2, fcw3) + fcb3, keepRate)

fcw4 = tf.Variable(tf.random_normal([256, 128], stddev=0.1), name="fullyConnectedWeights4")
fcb4 = tf.Variable(tf.random_normal([128], stddev=0.1), name="fullyConnectedBiases4")
fc4 = tf.nn.dropout(tf.matmul(fc3, fcw4) + fcb4, keepRate)

fcw5 = tf.Variable(tf.random_normal([128, 64], stddev=0.1), name="fullyConnectedWeights5")
fcb5 = tf.Variable(tf.random_normal([64], stddev=0.1), name="fullyConnectedBiases5")
fc5 = tf.nn.dropout(tf.matmul(fc4, fcw5) + fcb5, keepRate)

fcw6 = tf.Variable(tf.random_normal([64,30], stddev=0.1), name="fullyConnectedWeights6")
fcb6 = tf.Variable(tf.random_normal([30], stddev=0.1), name="fullyConnectedBiases6")
fc6 = tf.matmul(fc5, fcw6) + fcb6

loss = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(fc6, cor)), isZeroMuls))

trainStep = tf.train.AdamOptimizer().minimize(loss)
# trainStep = tf.train.AdadeltaOptimizer().minimize(loss)


images = []

modelPath = "checkpoint/model.ckpt"

# def getRandom(n):

with open("../data/training.csv") as training:
    file = csv.reader(training)
    next(file)
    for i, line in enumerate(file):
        # if i % 100 == 0: print("\r   ", i, end  = "\r")
        if i % 500 == 0: print(i)
        # print(len(line[-1].split()))
        nums = list(map(int, line[-1].split()))
        # print(line[:-1])
        ans = []
        zeroMuls = []
        for n in line[:-1]:
            if n == "":
                ans.append(0)
                zeroMuls.append(0)
                # print(i)
            else:
                ans.append(float(n))
                zeroMuls.append(1)
        images.append((nums, zeroMuls, ans))

with tf.Session() as sess:
    saver = tf.train.Saver()

    try:
        saver.restore(sess, modelPath)
    except:
        sess.run(tf.global_variables_initializer())

    i = 1
    while True:
        ans = []
        ii = []
        zs = []
        for (iii, z, c) in sample(images, 50):
            ii.append(iii)
            ans.append(c)
            zs.append(z)
        # print(len(ii) * len(ii[0]))
        # print(len(zs) * len(ii[0]))
        # print(len(ans))
        l, _ = sess.run((loss, trainStep), feed_dict={x: ii, cor: ans, isZeroMuls: zs, keepRate: 1})
        if i % 100 == 0:
            saver.save(sess, modelPath)
            print(str(i) + ":\t\t" + str(l) + "\t\tSaved")
        else:
            print(str(i) + ":\t\t" + str(l))
        i += 1
