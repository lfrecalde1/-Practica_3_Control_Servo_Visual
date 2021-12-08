#!/home/fer/.virtualenvs/cv/bin/python
import sys
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
from functions import *
import imutils
from imutils.object_detection import non_max_suppression

## Direcciones de donde se va sacar las Imagenes
path_o = '/home/fer/Control_servo_visual/Code/Practico_3.0/Pictures/'
path_o1 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Tools/'
path_o2 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Letters/'
path_o3 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Leafs/'
path_o4 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Persons/'

path_w = '/home/fer/Control_servo_visual/Code/Practico_3.0/Modificadas/'


def literal_1():
    print("Pregunta 1")
    imgs = data(path_o, 1)
    ## Pregunta_1
    trans = np.array([[25, 0, 0], [25, 0, 0]])
    angles = np.array([[0, 45, 0]])
    scales = np.array([[1, 1, 0.3]])

    pregunta_1(imgs, trans, angles, scales, path_w)

def literal_2():
    print("Pregunta 2")
    ## Pregunta 2
    imgs = data(path_o1, 1)
    Pregunta_2(imgs, path_w)

def literal_3():
    print("Pregunta 3")
    ## Pregunta 3
    imgs = data(path_o2, 1)
    img = imgs[0, :, :]
    Pregunt_3(img, path_w)


def literal_4():
    print("Pregunta 4")
    ## Pregunta 4
    imgs = data(path_o3, 1)
    pregunta_4(imgs, path_w)
    
def literal_5():
    print("Pregunta 5")
    imgs = data(path_o4, 1)
    pregunta_5(imgs, path_w)

def main():
    #literal_1()
    #literal_2()
    #literal_3()
    #literal_4()
    literal_5()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Press Ctrl-c to terminate the while statement")
        pass

