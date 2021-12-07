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
path_w = '/home/fer/Control_servo_visual/Code/Practico_3.0/Modificadas/'


def Pregunta_2(imgs, path):
    contador = 0
    for img in imgs:
        frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (0, 0, 0), (180, 255, 33))

        kernel1 = np.ones((2,2), np.uint8)  
        kernel2 = np.ones((6,6), np.uint8)  
        img_erosion = cv2.erode(frame_threshold, kernel1, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)

        M = cv2.moments(img_dilation)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # put text and highlight the center
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(img, "Centroide", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        name1 = "Pregunta_2_{}.png".format(contador)
        name2 = "Pregunta_2_normal_{}.png".format(contador)
        guardar(path, name1, img_dilation)
        guardar(path, name2, img)

        cv2.imshow("Original", img)
        cv2.imshow("Dst", img_dilation)
        cv2.waitKey(0)
        contador = contador + 1
    return None


def main():
    ## Lectura de las imagenes de prueba del sistema
    imgs = data(path_o1, 1)

    ## Uso de solo una iamgen para pruebas
    img = imgs[0,:,:]

    ## Pregunta_1
    trans = np.array([[25, 0, 0], [25, 0, 0]])
    angles = np.array([[0, 45, 0]])
    scales = np.array([[1, 1, 0.3]])

    #pregunta_1(imgs, trans, angles, scales, path_w)

    ## Pregunta 2
    Pregunta_2(imgs, path_w)

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

