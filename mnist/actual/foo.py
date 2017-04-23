import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


modelPath = "checkpoint/model.ckpt"

x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

xInp = tf.reshape(x, [-1, 28, 28, 1])

filter1 = tf.Variable(tf.random_normal([7,7,1,32], stddev=0.1), name="filter1")
biases1 = tf.Variable(tf.random_normal([32], stddev=0.1), name="biases1")
conv1 = tf.nn.relu(tf.nn.conv2d(xInp, filter1, [1,1,1,1], padding="VALID") + biases1)
pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], "SAME")

filter2 = tf.Variable(tf.random_normal([3,3,32,128], stddev=0.1), name="filter2")
biases2 = tf.Variable(tf.random_normal([128], stddev=0.1), name="biases2")
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2, [1,1,1,1], "VALID") + biases2)
pool2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], "SAME")

filter3 = tf.Variable(tf.random_normal([3,3,128, 256]), name="filter3")
biases3 = tf.Variable(tf.random_normal([256], stddev=0.1), name="biases3")
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, filter3, [1,1,1,1], "VALID") + biases3)

outFromConv = tf.reshape(conv3, [-1, 3*3*256])

keepRate = tf.placeholder(tf.float32)

fcw1 = tf.Variable(tf.random_normal([3*3*256, 2048], stddev=0.1), name="fullyConnectedWeights1")
fcb1 = tf.Variable(tf.random_normal([2048], stddev=0.1), name="fullyConnectedBiases1")
fc1 = tf.nn.dropout(tf.matmul(outFromConv, fcw1) + fcb1, keepRate)

fcw2 = tf.Variable(tf.random_normal([2048, 512], stddev=0.1), name="fullyConnectedWeights2")
fcb2 = tf.Variable(tf.random_normal([512], stddev=0.1), name="fullyConnectedBiases2")
fc2 = tf.nn.dropout(tf.matmul(fc1, fcw2) + fcb2, keepRate)

fcw3 = tf.Variable(tf.random_normal([512, 128], stddev=0.1), name="fullyConnectedWeights3")
fcb3 = tf.Variable(tf.random_normal([128], stddev=0.1), name="fullyConnectedBiases3")
fc3 = tf.nn.dropout(tf.matmul(fc2, fcw3) + fcb3, keepRate)

fcw4 = tf.Variable(tf.random_normal([128,10], stddev=0.1), name="fullyConnectedWeights4")
fcb4 = tf.Variable(tf.random_normal([10], stddev=0.1), name="fullyConnectedBiases4")
fc4 = tf.matmul(fc3, fcw4) + fcb4

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc4, y_))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropy)

correct = tf.cast(tf.equal(tf.argmax(fc4,1), tf.argmax(y_,1)), tf.float32)
accuracy = tf.reduce_mean(correct)

mnist = input_data.read_data_sets("data", one_hot=True)

with tf.Session() as sess:
    saver = tf.train.Saver()

    try:
        saver.restore(sess, modelPath)
    except:
        sess.run(tf.global_variables_initializer())

    for i in range(1000000):
        inx, iny_ = mnist.train.next_batch(200)
        if i % 1000 == 0:
            saver.save(sess, modelPath)
            print("Saved")
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: inx, y_: iny_, keepRate: 1.0})
            print(i, acc)
        else:
            sess.run(trainStep, feed_dict={x: inx, y_: iny_, keepRate: 0.5})

    saver.save(sess, modelPath)