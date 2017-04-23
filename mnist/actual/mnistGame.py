import pygame
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data
# import math
from random import sample

# colors
background = (0,0,0)

fill = (255,255,255)
border = (50,50,50)
borderWidth = 2

outline = (50,255,50)
outlineWidth = 5

mouseRadius = 0.65
mouse = (255, 50, 50)
mouseWidth = 3

textColor = (255,255,255)

dim = 28

down = False
popUp = False
text = ""
mousePos = (0,0)

screen = pygame.display.set_mode((500, 500), pygame.RESIZABLE)
# pixels = [[0 for _ in range(dim)] for _ in range(dim)]
pixels = [False for _ in range(dim*dim)]

def intersects(circle, radius, rect):
    cx, cy = circle
    rx, ry, rw, rh = rect
    r1 = pygame.Rect(cx - radius, cy - radius,radius * 2, radius * 2)
    r2 = pygame.Rect(rx, ry, rw, rh)
    return r1.colliderect(r2)


pygame.init()
########################################################################################################################
########################################################################################################################
########################################################################################################################




modelPath = "checkpoint/model.ckpt"

x = tf.placeholder(tf.float32, [None, 28*28])

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

res = tf.argmax(tf.nn.softmax(fc4), 1)

sess = tf.Session()

saver = tf.train.Saver()

try:
    saver.restore(sess, modelPath)
except:
    sess.run(tf.global_variables_initializer())


########################################################################################################################
########################################################################################################################
########################################################################################################################

# def intersects(circle, radius, rect):
#     def collision(rleft, rtop, width, height,  # rectangle definition
#                   center_x, center_y, radius):  # circle definition
#         """ Detect collision between a rectangle and circle. """
#
#         # complete boundbox of the rectangle
#         rright, rbottom = rleft + width / 2, rtop + height / 2
#
#         # bounding box of the circle
#         cleft, ctop = center_x - radius, center_y - radius
#         cright, cbottom = center_x + radius, center_y + radius
#
#         # trivial reject if bounding boxes do not intersect
#         if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
#             return False  # no collision possible
#
#         # check whether any point of rectangle is inside circle's radius
#         for x in (rleft, rleft + width):
#             for y in (rtop, rtop + height):
#                 # compare distance between circle's center point and each point of
#                 # the rectangle with the circle's radius
#                 if math.hypot(x - center_x, y - center_y) <= radius:
#                     return True  # collision detected
#
#         # check if center of circle is inside rectangle
#         if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
#             return True  # overlaid
#
#         return False  # no collision detected
#     cx, cy = circle
#     rx, ry, rw, rh = rect
#     return collision(rx, ry, rw, rh, cx, cy, radius)

def update():
    screen.fill(background)
    width = screen.get_width()
    height = screen.get_height()
    d = min(width, height)
    x0 = (width - d)/2
    y0 = (height - d)/2
    d /= dim

    for i in range(dim ** 2):
        x = i % 28
        y = i // 28

        x *= d
        y *= d

        if down:
            if intersects(mousePos, int(mouseRadius * d), (x0+ x, y0 + y, d, d)):
                pixels[i] = True

        if not(popUp) and pixels[i]:
            pygame.draw.rect(screen, fill, (x0+ x, y0 + y, d, d))

        pygame.draw.rect(screen, border, (x0 + x, y0 + y, d, d), borderWidth)

    pygame.draw.rect(screen, outline, (x0 + d*4,y0 +  d*4, d*(28-8), d*(28-8)), outlineWidth)

    if popUp:
        a = 1.7
        f = pygame.font.SysFont("Helvetica, Arial", int(d * 20 * a), True, False)
        tw, th = f.size(text)
        screen.blit(f.render(text, 0, textColor), (x0 + 14 * d - tw/2, y0 + 14 * d - (th * 0.9)/2))
    else:
        pygame.draw.circle(screen, mouse, mousePos, int(d * mouseRadius), mouseWidth)
        if down:
            pygame.draw.circle(screen, mouse, mousePos, int(d * mouseRadius))
    pygame.display.update()


update()
while True:
    for event in pygame.event.get():
        if event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            # print(event.size)
            update()
        elif event.type == pygame.QUIT:
            sess.close()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            mousePos = event.pos
            update()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            down = True
            update()
        elif event.type == pygame.MOUSEBUTTONUP:
            down = False
            update()

        # elif event.type == pygame.MOUSEBUTTONDOWN:
        #     x, y = pygame.mouse.get_pos()
        #     width = screen.get_width()
        #     height = screen.get_height()
        #     dim = max(width, height)
        #     x -= width/2
        #     x /= dim
        #     y -= height/2
        #     y /= dim
        #     x *= 2
        #     y *= 2
        #     points.append((x, y))
        #     print(x, y)
        #     update()
        elif event.type == pygame.KEYDOWN:
            if pygame.key.get_pressed()[pygame.K_BACKSPACE] == True:
                pixels = [False for _ in range(dim ** 2)]
            if pygame.key.get_pressed()[pygame.K_SPACE] == True:
                if not popUp:
                    r = sess.run(res, feed_dict={x: [list(map(lambda x: 1. if x else 0., pixels))], keepRate: 1.0})
                    text = str(r[0])
                popUp = not popUp
            if pygame.key.get_pressed()[pygame.K_ESCAPE] == True:
                popUp = False
            update()
