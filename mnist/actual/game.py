import pygame
import tensorflow as tf
import sys
from random import sample

# colors
coordsColor = (255,255,255)
coordsWidth = 5

pointColor = (200, 50,50)
pointRadius = 5

lineColor = (50, 200, 50)
lineWidth = 3

dev = 2

a = 0
g = 0


tfx = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable([[0.]])
b = tf.Variable([0.])
y = tf.matmul(tfx, w) + b
y_ = tf.placeholder(tf.float32, [None, 1])
crossEntropy = tf.reduce_mean(tf.square(y - y_))
trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)
res = tf.matmul(tf.constant([[-2.], [2.]]), w) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
def train():
    global a, g
    xs = []
    ys = []
    # for x__, y__ in sample(points, int(len(points)/2)):
    for x__, y__ in points:
        xs.append([x__])
        ys.append([y__])
    # print(xs,ys)
    r = sess.run([trainStep, res], feed_dict={tfx: xs, y_: ys})[1]
    # print(r)
    a = r[0][0]
    g = r[1][0]


pygame.init()

screen = pygame.display.set_mode((500, 500), pygame.RESIZABLE)
points = []

def toScr(point):
    x, y = point
    width = screen.get_width()
    height = screen.get_height()
    dim = max(width, height)
    x, y = point
    x *= dim / 2
    x += width/2
    y *= dim / 2
    y += height / 2
    return (int(x), int(y))

def drawPoint(screen, point):
    x, y = toScr(point)
    pygame.draw.circle(screen, pointColor, (x , y), pointRadius)

def drawCoords(screen):
    width, height = screen.get_size()
    pygame.draw.line(screen, coordsColor, (width/2, 0), (width/2, height), coordsWidth)
    pygame.draw.line(screen, coordsColor, (0, height/2), (width, height/2), coordsWidth)

def drawLine():
    # print(a,g)
    p1 = toScr((-dev, a))
    p2 = toScr((dev, g))
    # print(a,g)
    # print(p1, p2)
    pygame.draw.line(screen, lineColor, p1, p2, lineWidth)

def update():
    screen.fill((0,0,0))
    drawCoords(screen)
    drawLine()
    for point in points:
        drawPoint(screen, point)
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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            width = screen.get_width()
            height = screen.get_height()
            dim = max(width, height)
            x -= width/2
            x /= dim
            y -= height/2
            y /= dim
            x *= 2
            y *= 2
            points.append((x, y))
            # print(x, y)
            update()
        elif event.type == pygame.KEYDOWN:
            if pygame.key.get_pressed()[pygame.K_SPACE] == True:
                train()
            update()
        # speed[0] = -speed[0]
        # speed[1] = -speed[1]
