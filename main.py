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

def k_means(desired, actual):
    error = desired-actual
    distance = LA.norm(error, 2)
    return distance

def momentos(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    for k in range(0, hu_moments.shape[0]):
        hu_moments[k,0] = -1*np.sign(hu_moments[k,0])*np.log(np.abs(hu_moments[k,0]))

    return hu_moments


def Pregunt_3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    N1 = 35
    N2 = 29
    a = int((N1-1)/2)
    b = int((N2-1)/2)
    ## Creacion de la matriz igual
    aux1 = 17
    aux2 = 22

    ## Momento patron
    hu_desired1 = np.array([[  7.15268109],[ 17.69131612],[ 29.32038124],[ 29.20161667],[-60.49607098],[ 39.05400674],[ 58.47125494]])
    hu_desired2 = np.array([[  7.14670726],[ 17.73680959],[ 29.02479803],[ 29.72324143],[-59.23208216],[-40.32467434],[ 59.81848794]])

    desired1_distance = 9.94552596e-09
    desired2_distance = 7.77392093e-09
    desired_vector = np.array([[desired1_distance],[desired2_distance]])

    stepsx = np.arange(aux1, img.shape[0], 40)
    stepsy = np.arange(aux2, img.shape[1], 40)
    distance_vector = np.zeros((2,stepsx.shape[0]*stepsy.shape[0]))

    ## Bucle donde se ejecuta el algoritmo
    k2 = 0
    contador = 0
    for i in range(aux1, img.shape[0], 40):
        for j in range(aux2, img.shape[1], 40):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            hu_moments = momentos(A)
            distance1 = k_means(hu_desired1, hu_moments)
            distance2 = k_means(hu_desired2, hu_moments)

            aux1_distance = np.abs(distance1-desired1_distance)
            aux2_distance = np.abs(distance2-desired2_distance)

            if ((aux1_distance<=0.1) or(aux2_distance<0.1)):
                contador = contador+1

            distance_vector[0,k2] = distance1
            distance_vector[1,k2] = distance2
            cv2.imshow("Seccion", A)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            k2 =k2 +1

    resultado = distance_vector-desired_vector
    print("El numero d letras A es el siguiente")
    print(contador)
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
    Pregunt_3(img)

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

