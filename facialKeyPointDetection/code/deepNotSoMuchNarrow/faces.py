from argparse import _AppendAction

import tensorflow as tf
import csv
from random import sample
from sys import argv
from PIL import Image
import numpy as np
import os
import optparse


dim = 96

cor = tf.placeholder(tf.float32, [None])
mask = tf.placeholder(tf.bool, [None, 30])
# isZeroMuls = tf.placeholder(tf.float32, [None, 30])

x = tf.placeholder(tf.float32, [None, dim*dim])

xInp = tf.reshape(x, [-1, dim, dim, 1])

filter1 = tf.Variable(tf.truncated_normal([5,5,1,8], stddev=2), name="filter1")
biases1 = tf.Variable(tf.truncated_normal([8], stddev=0.0001), name="biases1")
conv1 = tf.nn.relu(tf.nn.conv2d(xInp, filter1, [1,1,1,1], padding="VALID") + biases1)

filter1_ = tf.Variable(tf.truncated_normal([5,5,8,16], stddev=2), name="filter1_")
biases1_ = tf.Variable(tf.truncated_normal([16], stddev=0.0001), name="biases1_")
conv1_ = tf.nn.relu(tf.nn.conv2d(conv1, filter1_, [1,1,1,1], padding="VALID") + biases1_)

pool1 = tf.nn.max_pool(conv1_, [1,2,2,1], [1,2,2,1], "SAME") # 44


filter2 = tf.Variable(tf.truncated_normal([5,5,16,32], stddev=2), name="filter2")
biases2 = tf.Variable(tf.truncated_normal([32], stddev=0.0001), name="biases2")
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2, [1,1,1,1], "VALID") + biases2)

filter2_ = tf.Variable(tf.truncated_normal([3,3,32,32], stddev=2), name="filter2_")
biases2_ = tf.Variable(tf.truncated_normal([32], stddev=0.0001), name="biases2_")
conv2_ = tf.nn.relu(tf.nn.conv2d(conv2, filter2_, [1,1,1,1], "VALID") + biases2_) # 38

pool2 = tf.nn.max_pool(conv2_, [1,2,2,1], [1,2,2,1], "SAME") # 19


filter3 = tf.Variable(tf.truncated_normal([4,4,32, 64], stddev=2), name="filter3")
biases3 = tf.Variable(tf.truncated_normal([64], stddev=0.0001), name="biases3")
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, filter3, [1,1,1,1], "VALID") + biases3)

filter3_ = tf.Variable(tf.truncated_normal([3,3,64, 64], stddev=2), name="filter3_")
biases3_ = tf.Variable(tf.truncated_normal([64], stddev=0.0001), name="biases3_")
conv3_ = tf.nn.relu(tf.nn.conv2d(conv3, filter3_, [1,1,1,1], "VALID") + biases3_) # 14

pool3 = tf.nn.max_pool(conv3_, [1,2,2,1], [1,2,2,1], "SAME") # 7


filter4 = tf.Variable(tf.truncated_normal([3,3,64, 128], stddev=2), name="filter4")
biases4 = tf.Variable(tf.truncated_normal([128], stddev=0.0001), name="biases4")
conv4 = tf.nn.relu(tf.nn.conv2d(pool3, filter4, [1,1,1,1], "VALID") + biases4)

filter4_ = tf.Variable(tf.truncated_normal([3,3,128, 256], stddev=2), name="filter4_")
biases4_ = tf.Variable(tf.truncated_normal([256], stddev=0.0001), name="biases4_")
conv4_ = tf.nn.relu(tf.nn.conv2d(conv4, filter4_, [1,1,1,1], "VALID") + biases4_)


filter5 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=2), name="filter5")
biases5 = tf.Variable(tf.truncated_normal([512], stddev=0.0001), name="biases5")
conv5 = tf.nn.relu(tf.nn.conv2d(conv4_, filter5, [1,1,1,1], "VALID") + biases5) # 1

# filter5_ = tf.Variable(tf.truncated_normal([3, 3, 512, 1024), name="filter5_")
# biases5_ = tf.Variable(tf.truncated_normal([1024], stddev=0.1), name="biases5_")
# conv5_ = tf.nn.relu(tf.nn.conv2d(conv5, filter5_, [1,1,1,1], "VALID") + biases5_) # 11

outFromConv = tf.reshape(conv5, [-1, 512])


keepRate = tf.placeholder(tf.float32)

# fcw0 = tf.Variable(tf.truncated_normal([1024, 512], stddev=2), name="fullyConnectedWeights1")
# fcb0 = tf.Variable(tf.truncated_normal([512], stddev=0.1), name="fullyConnectedBiases1")
# fc0 = tf.nn.dropout(tf.matmul(outFromConv, fcw0) + fcb0, keepRate)

fcw1 = tf.Variable(tf.truncated_normal([512, 256], stddev=2), name="fullyConnectedWeights1")
fcb1 = tf.Variable(tf.truncated_normal([256], stddev=0.0001), name="fullyConnectedBiases1")
fc1 = tf.nn.dropout(tf.matmul(outFromConv, fcw1) + fcb1, keepRate)

fcw2 = tf.Variable(tf.truncated_normal([256, 128], stddev=2), name="fullyConnectedWeights2")
fcb2 = tf.Variable(tf.truncated_normal([128], stddev=0.0001), name="fullyConnectedBiases2")
fc2 = tf.nn.dropout(tf.matmul(fc1, fcw2) + fcb2, keepRate)

fcw3 = tf.Variable(tf.truncated_normal([128, 128], stddev=2), name="fullyConnectedWeights3")
fcb3 = tf.Variable(tf.truncated_normal([128], stddev=0.0001), name="fullyConnectedBiases3")
fc3 = tf.nn.dropout(tf.matmul(fc2, fcw3) + fcb3, keepRate)

fcw3_ = tf.Variable(tf.truncated_normal([128, 64], stddev=2), name="fullyConnectedWeights3")
fcb3_ = tf.Variable(tf.truncated_normal([64], stddev=0.0001), name="fullyConnectedBiases3")
fc3_ = tf.nn.dropout(tf.matmul(fc3, fcw3_) + fcb3_, keepRate)

fcw4 = tf.Variable(tf.truncated_normal([64, 64], stddev=2), name="fullyConnectedWeights4")
fcb4 = tf.Variable(tf.truncated_normal([64], stddev=0.0001), name="fullyConnectedBiases4")
fc4 = tf.nn.dropout(tf.matmul(fc3_, fcw4) + fcb4, keepRate)

fcw5 = tf.Variable(tf.truncated_normal([64, 64], stddev=2), name="fullyConnectedWeights5")
fcb5 = tf.Variable(tf.truncated_normal([64], stddev=0.0001), name="fullyConnectedBiases5")
fc5 = tf.nn.dropout(tf.matmul(fc4, fcw5) + fcb5, keepRate)

fcw6 = tf.Variable(tf.truncated_normal([64,30], stddev=2), name="fullyConnectedWeights6")
fcb6 = tf.Variable(tf.truncated_normal([30], stddev=0.0001), name="fullyConnectedBiases6")
fc6 = tf.matmul(fc5, fcw6) + fcb6

# foo = tf.shape(tf.reshape(fc6, [-1]))
# foo1 = tf.shape(tf.reshape(cor, [-1]))

loss = tf.reduce_mean(tf.square(tf.abs(tf.subtract(tf.boolean_mask(tf.reshape(fc6, [-1]), tf.reshape(mask, [-1])), tf.reshape(cor, [-1])))))

trainStep = tf.train.AdamOptimizer().minimize(loss)
# trainStep = tf.train.AdadeltaOptimizer().minimize(loss)

# b = tf.Print(biases1, [biases1])

inpug = tf.Print(xInp, [xInp], "xInp", summarize=15)
c1 = tf.Print(inpug, [conv1], "conv1", summarize=15)
f1 = tf.Print(inpug, [filter1], "filter1", summarize=15)
b1 = tf.Print(inpug, [biases1], "biases1", summarize=15)
c2 = tf.Print(c1, [conv2], "conv2", summarize=15)
f2 = tf.Print(inpug, [filter2], "filter2", summarize=15)
b2 = tf.Print(inpug, [biases2], "biases2", summarize=15)
c3 = tf.Print(c2, [conv3], "conv3", summarize=15)
f3 = tf.Print(inpug, [filter3], "filter3", summarize=15)
b3 = tf.Print(inpug, [biases3], "biases3", summarize=15)
c4 = tf.Print(c2, [conv4], "conv4", summarize=15)
f4 = tf.Print(inpug, [filter4], "filter4", summarize=15)
b4 = tf.Print(inpug, [biases4], "biases4", summarize=15)
c5 = tf.Print(c2, [conv5], "conv5", summarize=15)
f5 = tf.Print(inpug, [filter5], "filter5", summarize=15)
b5 = tf.Print(inpug, [biases5], "biases5", summarize=15)
out = tf.Print((c2, c3, c4, c5), [outFromConv], "outfromconv", summarize=15)
f2_ = tf.Print((out), [fc2], "fc2", summarize=15)
f3_ = tf.Print(f2_, [fc3_], "fc3_", summarize=15)
f4_ = tf.Print(f3_, [fc4], "fc4", summarize=15)
f5_ = tf.Print(f4_, [fc5], "fc5", summarize=15)
f6 = tf.Print(f5_, [fc6], "fc6", summarize=15)
fb = tf.Print((f1, b1, f2, b2, f3, b3, f4, b4, f5, b5), [0], "")




images = []

modelPath = "checkpoint/model.ckpt"

# def getRandom(n):

def processImages(args, sess, prefix="recognized_", show=True, save=True):
    global x, cor, mask, keepRate
    imageNames = []
    images = []
    outImages = []
    for arg in args:
        try:
            image = Image.open(arg)
            if image.height == image.width == 96:
                image = image.convert("RGB")
                # print(image.getdata())
                # image.show()
                lImage = list(image.getdata())
                # print(lImage)
                # print(list(map(lambda rgb: (rgb[0] + rgb[1] + rgb[2])/(255*3), lImage)))
                images.append(list(map(lambda rgb: ((rgb[0] + rgb[1] + rgb[2])/(255*3))*2 - 1, lImage)))
                outImages.append(list(lImage))
                parts = os.path.split(arg)
                imageNames.append(parts[0] + "/" + prefix + parts[1])
        except:
            pass
    # print(len(images))
    if len(images) > 0:

        def group(sequence, chunk_size):
            return list(zip(*[iter(sequence)] * chunk_size))

        re = []
        green = [142, 250, 0]

        def paste(x, y, im):
            i = y * 96 + x
            if i < len(im) and i >= 0:
                im[i] = green


        re = sess.run(fc6, feed_dict={x: images, cor: [], mask: [[True]*30], keepRate: 1})
        # print(re)
        for i, rs in enumerate(re):
            image = outImages[i]
            name = imageNames[i]

            for (x_, y) in group(rs, 2):
                def back(n):
                    return int(((n + 1 ) / 2 ) * dim)
                x_, y = back(x_), back(y)
                things = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in things:
                    paste(x_+dx, y+dy, image)

            c = np.asarray(image)
            c = c.reshape([96,96, 3]).astype("uint8")

            # print(i)

            sImage = Image.fromarray(c, "RGB")
            if save:
                sImage.save(name)
            if show:
                sImage.show()


parser = optparse.OptionParser()
parser.add_option("-t", "--train", action="store_true", dest="train", default=False)
parser.add_option("-F", "--testfile", type="string", dest="testfile")
parser.add_option("-f", "--file", type="string", dest="file")

(opt, _) = parser.parse_args()

if opt.train:
    with open("../data/training.csv") as training:
        file = csv.reader(training)
        next(file)
        for i, line in enumerate(file):
            # if i % 100 == 0: print("\r   ", i, end  = "\r")
            if i % 500 == 0: print(i)
            # print(len(line[-1].split()))
            nums = list(map(lambda y: (int(y) / 255)*2 - 1, line[-1].split()))
            # print(line[:-1])
            ans = []
            zeroMuls = []
            for n in line[:-1]:
                if n == "":
                    # ans.append(0)
                    zeroMuls.append(False)
                    # print(i)
                else:
                    ans.append((float(n)/dim)*2 - 1 )
                    zeroMuls.append(True)
            images.append((nums, zeroMuls, ans))

    with tf.Session() as sess:
        saver = tf.train.Saver()

        try:
            saver.restore(sess, modelPath)
        except:
            sess.run(tf.global_variables_initializer())

        i = 126201
        while True:
            ansz = []
            ii = []
            zs = []
            for (iii, z, c) in sample(images, 50):
                ii.append(iii)
                ansz.append(c)
                zs.append(z)
            # print(len(ii) * len(ii[0]))
            # print(len(zs) * len(ii[0]))
            # print(len(ansz))
            # print(ii, ansz, zs)
            l, _ = sess.run((loss, trainStep), feed_dict={x: ii, cor: [y for yy in ansz for y in yy], mask: zs, keepRate: 0.5})
            assert (l < 900000000000000000), l
            # print(l)
            # print(f, f1)
            if i % 100 == 0:
                saver.save(sess, modelPath)
                print(str(i) + ":\t\t" + str(l) + "\t\tSaved")
                processImages(argv[1:], sess, prefix="test" + str(i) + "_", show=False)
                with open("log.txt", "a") as log:
                    log.write(str(l) + "\n")
            elif i % 5 == 0:
                print(str(i) + ":\t\t" + str(l))
                # print(list(zip(cqr[0], fqr[0])))
            i += 1
else:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelPath)
        processImages(argv[1:], sess)