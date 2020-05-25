import os
import cv2
import numpy as np

URL_imagenes = "./data_Hungaro/secuencia_jugador/"
URL_detecciones = "./data_Hungaro/detecciones_jugador.csv"
data = os.listdir(URL_imagenes)
data.sort()

detecciones = np.genfromtxt(URL_detecciones, delimiter=",")

colores = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

cont = 0

def calcular_matriz_coste(d1, d0):
    m_cost = np.zeros((d1.shape[0], d1.shape[1]))
    for i in range(len(d1[0])):
        for j in range(len(d0[0])):
            c0 = np.array([d0[j][0] + (d0[j][2] / 2), d0[j][1] + (d0[j][3] / 2)])
            c1 = np.array([d1[i][0] + (d1[i][2] / 2), d0[j][1] + (d0[j][3] / 2)])
            m_cost[j][i] = int(np.linalg.norm(d1[i] - d0[j]))

    return m_cost

def comprobar_matriz(costes):
    seleccionados_y = []
    cruces = []

    for i in range(costes.shape[0]):
        det = np.where(costes[:, i] == 0)
        if det[0].shape[0] == 1:
            aux = np.where(seleccionados_y == det[0])
            if len(aux[0]) == 0:
                cruces.append((i, det[0][0]))
                seleccionados_y.append(det[0][0])
            else:
                return False, cruces
        else:
            for j in range(det[0].shape[0]):
                aux = np.where(seleccionados_y == det[0][j])
                if len(aux[0]) == 0:
                    cruces.append((i, det[0][j]))
                    seleccionados_y.append(det[0][j])
                    break
    if len(cruces) == costes.shape[0]:
        return True, cruces
    else:
        return False, cruces

def calcular_asociaciones(costes):
    # Encontrar el minimo de cada fila y restarlo
    for i in range(costes.shape[0]):
        costes[i] = costes[i] - np.amin(costes[i])
    # Encontrar el minimo de cada columna y restarlo
    for j in range(costes.shape[1]):
        costes[:, j] = costes[:, j] - np.amin(costes[:, j])

    fin, cruces = comprobar_matriz(costes.copy())

    if fin:
        return cruces
    else:
        print("Hay que continuar")


def pintar_detecciones(detecciones, colores, asociaciones, frame):
    colores_nuevos = []
    for i in range(len(detecciones)):
        p1 = (int(detecciones[i][0]), int(detecciones[i][1]))
        p2 = (int(detecciones[i][0] + detecciones[i][2]), int(detecciones[i][1] + detecciones[i][3]))
        color = colores[asociaciones[i][1]]
        cv2.rectangle(frame, p1, p2, color, 2)
        colores_nuevos.append(color)
    return frame, colores_nuevos


for img in data:
    frame = cv2.imread(os.path.join(URL_imagenes, img))

    d1 = np.array([detecciones[cont][0:4],
                   detecciones[cont][4:8],
                   detecciones[cont][8:12],
                   detecciones[cont][12:17]])

    if cont == 0:
        matriz_coste = calcular_matriz_coste(d1, d1)
    else:
        d0 = np.array([detecciones[cont - 1][0:4],
                       detecciones[cont - 1][4:8],
                       detecciones[cont - 1][8:12],
                       detecciones[cont - 1][12:17]])
        matriz_coste = calcular_matriz_coste(d1, d0)

    asociaciones = calcular_asociaciones(matriz_coste.copy())
    #coste = calcular_coste(matriz_coste.copy(), asociaciones)

    if asociaciones != None:
        frame, nuevos_colores = pintar_detecciones(d1, colores, asociaciones, frame.copy())
        colores = nuevos_colores
        cv2.imshow("Hungaro", frame)
    cont = cont + 1

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
