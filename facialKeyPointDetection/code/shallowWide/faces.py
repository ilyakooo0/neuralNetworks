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

filter1_ = tf.Variable(tf.random_normal([7,7,16,32], stddev=0.1), name="filter1_")
biases1_ = tf.Variable(tf.random_normal([32], stddev=0.1), name="biases1_")
conv1_ = tf.nn.relu(tf.nn.conv2d(conv1, filter1_, [1,1,1,1], padding="VALID") + biases1_) # 84

# pool1 = tf.nn.max_pool(conv1_, [1,2,2,1], [1,2,2,1], "SAME")


filter2 = tf.Variable(tf.random_normal([6,6,32,64], stddev=0.1), name="filter2")
biases2 = tf.Variable(tf.random_normal([64], stddev=0.1), name="biases2")
conv2 = tf.nn.relu(tf.nn.conv2d(conv1_, filter2, [1,1,1,1], "VALID") + biases2)

filter2_ = tf.Variable(tf.random_normal([6,6,64,128], stddev=0.1), name="filter2_")
biases2_ = tf.Variable(tf.random_normal([128], stddev=0.1), name="biases2_")
conv2_ = tf.nn.relu(tf.nn.conv2d(conv2, filter2_, [1,1,1,1], "VALID") + biases2_) # 74

pool2 = tf.nn.max_pool(conv2_, [1,2,2,1], [1,2,2,1], "SAME") # 37


filter3 = tf.Variable(tf.random_normal([6,6,128, 256]), name="filter3")
biases3 = tf.Variable(tf.random_normal([256], stddev=0.1), name="biases3")
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, filter3, [1,1,1,1], "VALID") + biases3)

filter3_ = tf.Variable(tf.random_normal([5,5,256, 512]), name="filter3_")
biases3_ = tf.Variable(tf.random_normal([512], stddev=0.1), name="biases3_")
conv3_ = tf.nn.relu(tf.nn.conv2d(conv3, filter3_, [1,1,1,1], "VALID") + biases3_) # 28

pool3 = tf.nn.max_pool(conv3_, [1,2,2,1], [1,2,2,1], "SAME") # 14


# filter4 = tf.Variable(tf.random_normal([5,5,512, 1024]), name="filter4")
# biases4 = tf.Variable(tf.random_normal([1024], stddev=0.1), name="biases4")
# conv4 = tf.nn.relu(tf.nn.conv2d(pool3, filter4, [1,1,1,1], "VALID") + biases4) # 10
# pool4 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], "SAME") # 5
#
# # filter4_ = tf.Variable(tf.random_normal([2,2,1024, 2048]), name="filter4_")
# # biases4_ = tf.Variable(tf.random_normal([2048], stddev=0.1), name="biases4_")
# # conv4_ = tf.nn.relu(tf.nn.conv2d(pool4, filter4_, [1,1,1,1], "VALID") + biases4_) # 4
#
#
# filter5 = tf.Variable(tf.random_normal([5, 5, 1024, 2048]), name="filter5")
# biases5 = tf.Variable(tf.random_normal([2048], stddev=0.1), name="biases5")
# conv5 = tf.nn.relu(tf.nn.conv2d(pool4, filter5, [1,2,2,1], "VALID") + biases5)
#
# outFromConv = tf.reshape(conv5, [-1, 2048])

filter_ = tf.Variable(tf.random_normal([14,14,512, 1024]), name="filter_")
biases_ = tf.Variable(tf.random_normal([1024], stddev=0.1), name="biases_")
conv_ = tf.nn.relu(tf.nn.conv2d(pool3, filter_, [1,1,1,1], "VALID") + biases_)

# pool_ = tf.nn.max_pool(conv_, [1,2,2,1], [1,2,2,1], "SAME")

outFromConv = tf.reshape(conv_, [-1, 1024])

keepRate = tf.placeholder(tf.float32)

# fcw0 = tf.Variable(tf.random_normal([4096, 2048], stddev=0.1), name="fullyConnectedWeights1")
# fcb0 = tf.Variable(tf.random_normal([2048], stddev=0.1), name="fullyConnectedBiases1")
# fc0 = tf.nn.dropout(tf.matmul(outFromConv, fcw0) + fcb0, keepRate)
#
# fcw1 = tf.Variable(tf.random_normal([2048, 1024], stddev=0.1), name="fullyConnectedWeights1")
# fcb1 = tf.Variable(tf.random_normal([1024], stddev=0.1), name="fullyConnectedBiases1")
# fc1 = tf.nn.dropout(tf.matmul(outFromConv, fcw1) + fcb1, keepRate)

fcw2 = tf.Variable(tf.random_normal([1024, 512], stddev=0.1), name="fullyConnectedWeights2")
fcb2 = tf.Variable(tf.random_normal([512], stddev=0.1), name="fullyConnectedBiases2")
fc2 = tf.nn.dropout(tf.matmul(outFromConv, fcw2) + fcb2, keepRate)

# fcw3 = tf.Variable(tf.random_normal([8192, 2048], stddev=0.1), name="fullyConnectedWeights3")
# fcb3 = tf.Variable(tf.random_normal([2048], stddev=0.1), name="fullyConnectedBiases3")
# fc3 = tf.nn.dropout(tf.matmul(fc2, fcw3) + fcb3, keepRate)

fcw3_ = tf.Variable(tf.random_normal([512, 256], stddev=0.1), name="fullyConnectedWeights3")
fcb3_ = tf.Variable(tf.random_normal([256], stddev=0.1), name="fullyConnectedBiases3")
fc3_ = tf.nn.dropout(tf.matmul(fc2, fcw3_) + fcb3_, keepRate)

fcw4 = tf.Variable(tf.random_normal([256, 128], stddev=0.1), name="fullyConnectedWeights4")
fcb4 = tf.Variable(tf.random_normal([128], stddev=0.1), name="fullyConnectedBiases4")
fc4 = tf.nn.dropout(tf.matmul(fc3_, fcw4) + fcb4, keepRate)

fcw5 = tf.Variable(tf.random_normal([128, 64], stddev=0.1), name="fullyConnectedWeights5")
fcb5 = tf.Variable(tf.random_normal([64], stddev=0.1), name="fullyConnectedBiases5")
fc5 = tf.nn.dropout(tf.matmul(fc4, fcw5) + fcb5, keepRate)

fcw6 = tf.Variable(tf.random_normal([64,30], stddev=0.1), name="fullyConnectedWeights6")
fcb6 = tf.Variable(tf.random_normal([30], stddev=0.1), name="fullyConnectedBiases6")
fc6 = tf.matmul(fc5, fcw6) + fcb6

loss = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(fc6, cor)), isZeroMuls))

trainStep = tf.train.AdamOptimizer().minimize(loss)
# trainStep = tf.train.AdadeltaOptimizer().minimize(loss)


inpug = tf.Print(xInp, [xInp], "xInp", summarize=15)
c1 = tf.Print(inpug, [conv1], "conv1", summarize=15)
c2 = tf.Print(c1, [conv2], "conv2", summarize=15)
c3 = tf.Print(c2, [conv3], "conv3", summarize=15)
out = tf.Print(c2, [outFromConv], "outfromconv", summarize=15)
f2 = tf.Print((c3, out), [fc2], "fc2", summarize=15)
f3 = tf.Print(f2, [fc3_], "fc3_", summarize=15)
f4 = tf.Print(f3, [fc4], "fc4", summarize=15)
f5 = tf.Print(f4, [fc5], "fc5", summarize=15)
f6 = tf.Print(f5, [fc6], "fc6", summarize=15)





images = []

modelPath = "checkpoint/model.ckpt"

# def getRandom(n):

with open("data/training.csv") as training:
    file = csv.reader(training)
    next(file)
    for i, line in enumerate(file):
        # if i % 100 == 0: print("\r   ", i, end  = "\r")
        if i % 500 == 0: print(i)
        # print(len(line[-1].split()))
        nums = list(map(lambda y: float(y)/255, line[-1].split()))
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
        l, fg,  _ , _= sess.run((loss, fc6, trainStep, f6), feed_dict={x: ii, cor: ans, isZeroMuls: zs, keepRate: 0.5})
        print(l, fg)
        if i % 1000 == 0:
            saver.save(sess, modelPath)
            print(str(i) + ":\t\t" + str(l) + "\t\tSaved")
        elif i % 100 == 0:
            print(str(i) + ":\t\t" + str(l))
        i += 1
