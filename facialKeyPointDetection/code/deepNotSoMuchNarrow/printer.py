import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

dim = 96

filter1 = tf.Variable(tf.truncated_normal([5,5,1,8], stddev=5), name="filter1")
biases1 = tf.Variable(tf.truncated_normal([8], stddev=0.001), name="biases1")

filter1_ = tf.Variable(tf.truncated_normal([5,5,8,16], stddev=5), name="filter1_")
biases1_ = tf.Variable(tf.truncated_normal([16], stddev=0.001), name="biases1_")

filter2 = tf.Variable(tf.truncated_normal([5,5,16,32], stddev=5), name="filter2")
biases2 = tf.Variable(tf.truncated_normal([32], stddev=0.001), name="biases2")

filter2_ = tf.Variable(tf.truncated_normal([3,3,32,32], stddev=5), name="filter2_")
biases2_ = tf.Variable(tf.truncated_normal([32], stddev=0.001), name="biases2_")

filter3 = tf.Variable(tf.truncated_normal([4,4,32, 64], stddev=5), name="filter3")
biases3 = tf.Variable(tf.truncated_normal([64], stddev=0.001), name="biases3")

filter3_ = tf.Variable(tf.truncated_normal([3,3,64, 64], stddev=5), name="filter3_")
biases3_ = tf.Variable(tf.truncated_normal([64], stddev=0.001), name="biases3_")

filter4 = tf.Variable(tf.truncated_normal([3,3,64, 128], stddev=5), name="filter4")
biases4 = tf.Variable(tf.truncated_normal([128], stddev=0.001), name="biases4")

filter4_ = tf.Variable(tf.truncated_normal([3,3,128, 256], stddev=5), name="filter4_")
biases4_ = tf.Variable(tf.truncated_normal([256], stddev=0.001), name="biases4_")

filter5 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=5), name="filter5")
biases5 = tf.Variable(tf.truncated_normal([512], stddev=0.001), name="biases5")

# filter5_ = tf.Variable(tf.truncated_normal([3, 3, 512, 1024), name="filter5_")
# biases5_ = tf.Variable(tf.truncated_normal([1024], stddev=0.1), name="biases5_")

# fcw0 = tf.Variable(tf.truncated_normal([1024, 512], stddev=5), name="fullyConnectedWeights1")
# fcb0 = tf.Variable(tf.truncated_normal([512], stddev=0.1), name="fullyConnectedBiases1")
# fc0 = tf.nn.dropout(tf.matmul(outFromConv, fcw0) + fcb0, keepRate)

fcw1 = tf.Variable(tf.truncated_normal([512, 256], stddev=5), name="fullyConnectedWeights1")
fcb1 = tf.Variable(tf.truncated_normal([256], stddev=0.001), name="fullyConnectedBiases1")

fcw2 = tf.Variable(tf.truncated_normal([256, 128], stddev=5), name="fullyConnectedWeights2")
fcb2 = tf.Variable(tf.truncated_normal([128], stddev=0.001), name="fullyConnectedBiases2")

fcw3 = tf.Variable(tf.truncated_normal([128, 128], stddev=5), name="fullyConnectedWeights3")
fcb3 = tf.Variable(tf.truncated_normal([128], stddev=0.001), name="fullyConnectedBiases3")

fcw3_ = tf.Variable(tf.truncated_normal([128, 64], stddev=5), name="fullyConnectedWeights3")
fcb3_ = tf.Variable(tf.truncated_normal([64], stddev=0.001), name="fullyConnectedBiases3")

fcw4 = tf.Variable(tf.truncated_normal([64, 64], stddev=5), name="fullyConnectedWeights4")
fcb4 = tf.Variable(tf.truncated_normal([64], stddev=0.001), name="fullyConnectedBiases4")

fcw5 = tf.Variable(tf.truncated_normal([64, 64], stddev=5), name="fullyConnectedWeights5")
fcb5 = tf.Variable(tf.truncated_normal([64], stddev=0.001), name="fullyConnectedBiases5")

fcw6 = tf.Variable(tf.truncated_normal([64,30], stddev=5), name="fullyConnectedWeights6")
fcb6 = tf.Variable(tf.truncated_normal([30], stddev=0.001), name="fullyConnectedBiases6")

# foo = tf.shape(tf.reshape(fc6, [-1]))
# foo1 = tf.shape(tf.reshape(cor, [-1]))
# trainStep = tf.train.AdadeltaOptimizer().minimize(loss)

# b = tf.Print(biases1, [biases1])

f1 = tf.Print(filter1, [filter1], "filter1", summarize=15)
b1 = tf.Print(filter1, [biases1], "biases1", summarize=15)
f2 = tf.Print(filter1, [filter2], "filter2", summarize=15)
b2 = tf.Print(filter1, [biases2], "biases2", summarize=15)
f3 = tf.Print(filter1, [filter3], "filter3", summarize=15)
b3 = tf.Print(filter1, [biases3], "biases3", summarize=15)
f4 = tf.Print(filter1, [filter4], "filter4", summarize=15)
b4 = tf.Print(filter1, [biases4], "biases4", summarize=15)
f5 = tf.Print(filter1, [filter5], "filter5", summarize=15)
b5 = tf.Print((f1, b1, f2, b2, f3, b3, f4, b4, f5), [biases5], "biases5", summarize=15)



images = []

modelPath = "checkpoint/model.ckpt"

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, modelPath)
    sess.run(b5)