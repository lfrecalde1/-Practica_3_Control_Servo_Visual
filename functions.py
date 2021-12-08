#!/home/fer/.virtualenvs/cv/bin/python
import sys
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA

def data(path,aux=1):                                                                                                                                                                                                                      
    images = []                                                                                                         
    index = os.listdir(path)                                                                                            
    index.sort()                                                                                                        
    for img in index:                                                                                                   
                                                                                                                        
        pictures = cv2.imread(os.path.join(path,img), aux)                                                              
        images.append([pictures])                                                                                       
        if aux==1:                                                                                                          
            img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1],pictures.shape[2])
        else:                                                                                                               
            img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1])                  
    return img  

def guardar(direccion, name, new):
    cv2.imwrite(os.path.join(direccion, name), new)
    return None

def refill(img):
    copy = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    copy = copy.astype("uint8")
    cv2.floodFill((copy), mask, (0,0), 255)
    copy_inv = cv2.bitwise_not(copy)
    salida = img | copy_inv
    return salida


def rotate(img, angles = 0, center = None, t = [0, 0], scales = 1.0):

    ## Obtencion del angulo de rotacion del sistema
    x = t[0]
    y = t[1]

    scale =scales
    angle = angles

    ## definicion de la dimesiones de la imagen
    (h, w) = img.shape[:2]

    ## Definicion del centro de la imagen 
    if center is None:
        center = (w /2)+x, (h /2)+y

    translacion = np.float64([[1, 0, x], [0, 1, y]])

    ## Seccion de la rotacion de la iamgen 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M_a = np.asarray([M[0, :], M[1, :], [0, 0, 1]])

    ## Aument traslation and rotation for compplete system

    translacion = np.array([translacion[0,:], translacion[1,:], [0, 0, 1]])

    ## tranformation General

    H = translacion@M_a

    rotate =  cv2.warpAffine(img, H[0:2,:], (w,h))
    return rotate

def corners(src_gray):

    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    ## cambio de variale
    src_gray = cv2.cvtColor(src_gray, cv2.COLOR_BGR2GRAY)
    src = np.array(src_gray, dtype = np.float32)

    # Detecting corners
    dst = cv2.cornerHarris(src, blockSize, apertureSize, k)

    dst = cv2.dilate(dst, None)

    src_gray[dst>0.2*dst.max()] = 255

    return src_gray 

def pregunta_1(imgs, tran, angle, scale, path):
    contador = 0
    ## Bucle analisis
    for img in imgs:
        for k in range(0,tran.shape[1]):
            dst = rotate(img, angles= angle[0, k], center = None, t = tran[:,k], scales = scale[0, k])
            corner_1 = corners(img)
            corner_2 = corners(dst)
            name1 = "Pregunta_1_{}_{}.png".format(contador, k)
            name2 = "Pregunta_1_normal_{}_{}.png".format(contador, k)
            guardar(path, name1, corner_2)
            guardar(path, name2, corner_1)
            cv2.imshow("Normal Corner", corner_1)
            cv2.imshow("Dst Corner", corner_2)
            cv2.waitKey(0)
        contador = contador+1
    return None


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
    return None
