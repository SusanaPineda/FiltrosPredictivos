import os
import cv2
import numpy as np

from Kalman import kalman

URL_detecciones = "./dataKalman/detecciones/ball_20.csv"
URL_imagenes = "./dataKalman/images/"

detecciones = np.genfromtxt(URL_detecciones, delimiter=",")

imagenes = os.listdir(URL_imagenes)
imagenes.sort()

dk = 0.1

q = 0.001
r = 0.1

A = np.array([[1, 0, dk, 0],
              [0, 1, 0, dk],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

X_ini = np.zeros((4, 1))

P_ini = np.eye(4)

R = np.array([[r, 0],
             [0, r]])

Q = np.array([[(dk ** 4) / 4, 0, (dk ** 3) / 2, 0],
              [0, (dk ** 4) / 4, 0, (dk ** 3) / 2],
              [(dk ** 3) / 2, 0, dk ** 2, 0],
              [0, (dk ** 3) / 2, 0, dk ** 2]]) * q ** 2

B = np.array([[(dk ** 2) / 2, 0],
              [(dk ** 2) / 2, 0],
              [dk, 0],
              [0, dk]])

klm = kalman(A, H, X_ini, P_ini, R, Q, B)

for i in range(len(imagenes)):
    deteccion = detecciones[i].astype(int)
    img = cv2.imread(os.path.join(URL_imagenes, imagenes[i]))

    # medida ruidosa en rojo
    p1_det = (deteccion[0], deteccion[1])
    p2_det = (deteccion[0]+deteccion[2], deteccion[1]+deteccion[3])
    cv2.rectangle(img, p1_det, p2_det, (0, 0, 255), 2)

    (x_pred, y_pred, xv_pred, yv_pred) = klm.predict()

    # prediccion en azul
    p1_pred = (int(x_pred[0]), int(y_pred[0]))
    p2_pred = (int(x_pred[0]+deteccion[2]), int(y_pred[0]+deteccion[3]))
    cv2.rectangle(img, p1_pred, p2_pred, (255, 0, 0), 2)

    act = np.array([[deteccion[0]+(deteccion[2]/2)], [deteccion[1]+(deteccion[3]/2)]])
    (x_corr, y_corr, xv_corr, yv_corr) = klm.correct(act)

    # corrección en verde
    p1_corr = (int(x_corr[0]), int(y_corr[0]))
    p2_corr = (int(x_corr[0] + deteccion[2]), int(y_corr[0] + deteccion[3]))
    cv2.rectangle(img, p1_corr, p2_corr, (0, 255, 0), 2)

    cv2.imshow("Kalman", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()