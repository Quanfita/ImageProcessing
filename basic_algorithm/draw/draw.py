import cv2
import numpy as np

def drawPoint(canvas,x,y):
    canvas[y,x] = 0

def drawLine(canvas,x1,y1,x2,y2):
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    xi, yi = x1, y1
    sx, sy = 1 if (x2 - x1) > 0 else -1, 1 if (y2 - y1) > 0 else -1
    pi = 2*dy - dx

    while xi != x2 + 1:
        if pi < 0:
            pi += 2 * dy
        else:
            pi += 2 * dy - 2 * dx
            yi += 1 * sy
        drawPoint(canvas,xi,yi)
        xi += 1 * sx

def drawCircle(canvas,x,y,r):
    x0, y0 = x, y
    xi = 0
    yi = r
    pi = 5/4 - r

    while xi <= yi:
        if pi < 0:
            pi += 2 * (xi + 1) + 1
        else:
            pi += 2 * (xi + 1) + 1 - 2 * (yi - 1)
            yi -= 1
        drawPoint(canvas,xi+x0,yi+y0)
        drawPoint(canvas,-xi+x0,yi+y0)
        drawPoint(canvas,xi+x0,-yi+y0)
        drawPoint(canvas,-xi+x0,-yi+y0)
        xi += 1
    
    xi = r
    yi = 0
    pi = 5/4 - r

    while not (xi == yi+1 or xi == yi):
        if pi < 0:
            pi += 2 * (yi + 1) + 1
        else:
            pi += 2 * (yi + 1) + 1 - 2 * (xi - 1)
            xi -= 1
        drawPoint(canvas,xi+x0,yi+y0)
        drawPoint(canvas,-xi+x0,yi+y0)
        drawPoint(canvas,xi+x0,-yi+y0)
        drawPoint(canvas,-xi+x0,-yi+y0)
        yi += 1

def drawEllipse(canvas,x,y,rx,ry):
    x0, y0 = x, y
    xi, yi = 0, ry
    rx2 = rx ** 2
    ry2 = ry ** 2
    p1i = ry2 - rx2 * ry + rx2 / 4
    while 2*ry2*xi < 2*rx2*yi:
        if p1i < 0:
            p1i += 2 * ry2 * (xi + 1) + ry2
        else:
            p1i += 2 * ry2 * (xi + 1) - 2* rx2 * (yi - 1) + ry2
            yi -= 1
        drawPoint(canvas,xi+x0,yi+y0)
        drawPoint(canvas,-xi+x0,yi+y0)
        drawPoint(canvas,xi+x0,-yi+y0)
        drawPoint(canvas,-xi+x0,-yi+y0)
        xi += 1
    xi -= 1
    p2i = ry2 * (xi + .5) ** 2 + rx2 * (yi - 1) ** 2 - rx2 * ry2
    while yi >= 0:
        if p2i > 0:
            p2i += -2 * rx2 * (yi - 1) + rx2
        else:
            p2i += 2 * ry2 * (xi + 1) - 2 * rx2 * (yi - 1) + rx2
            xi += 1
        drawPoint(canvas,xi+x0,yi+y0)
        drawPoint(canvas,-xi+x0,yi+y0)
        drawPoint(canvas,xi+x0,-yi+y0)
        drawPoint(canvas,-xi+x0,-yi+y0)
        yi -= 1


if __name__ == '__main__':
    canvas = np.ones([1000,1000],dtype=np.uint8) * 255
    drawLine(canvas,800,100,100,600)
    cv2.imwrite('line.png',canvas)
    canvas = np.ones([1000,1000],dtype=np.uint8) * 255
    drawCircle(canvas,500,500,300)
    cv2.imwrite('circle.png',canvas)
    canvas = np.ones([1000,1000],dtype=np.uint8) * 255
    drawEllipse(canvas,500,500,100,200)
    cv2.imwrite('ellipse.png',canvas)