#!/home/fer/.virtualenvs/cv/bin/python
import sys
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
from functions import *

## Direcciones de donde se va sacar las Imagenes
path_o = '/home/fer/Control_servo_visual/Code/Practico_3.0/Pictures/'
path_o1 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Tools/'
path_o2 = '/home/fer/Control_servo_visual/Code/Practico_3.0/Letters/'

path_w = '/home/fer/Control_servo_visual/Code/Practico_3.0/Modificadas/'


def main():
    ## Lectura de las imagenes de prueba del sistema
    #imgs = data(path_o, 1)


    ## Pregunta_1
    #trans = np.array([[25, 0, 0], [25, 0, 0]])
    #angles = np.array([[0, 45, 0]])
    #scales = np.array([[1, 1, 0.3]])

    #pregunta_1(imgs, trans, angles, scales, path_w)

    ## Pregunta 2
    #imgs = data(path_o1, 1)

    #Pregunta_2(imgs, path_w)

    ## Pregunta 4
    imgs = data(path_o2, 1)
    img = imgs[0, :, :]
    cv2.imshow("Letras", img)
    cv2.waitKey(0)

    ## Prueba Funcion 
    #dif, prueba, Mascara = match(img)

    ## Grafica de la Imagen y de la plantilla
    #show(prueba, dif)
    #show(img, prueba)

    #bucle(img)


    

    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Press Ctrl-c to terminate the while statement")
        pass

