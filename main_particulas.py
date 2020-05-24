import cv2
import os
import numpy as np
from random import randint, uniform, random

def inicializar_particulas(w, h, n, w_img, h_img):
    particulas = []
    for i in range(n):
        x = randint(0, w_img-w)
        y = randint(0, h_img-h)
        particulas.append(np.array([x, y, h, w, 0, 0]))

    return particulas


def evaluacion(frame, particulas):
    for p in particulas:
        roi = frame[p[1]:p[1]+p[2], p[0]:p[0]+p[3]]
        peso = np.count_nonzero(roi)
        p[4] = peso
    return particulas


def estimacion(particulas, pos_actual):
    global inicializado
    aux_index = -1
    aux_peso = 0
    for i in range(len(particulas)):
        p = particulas[i]
        if p[4] > aux_peso:
            aux_peso = p[4]
            aux_index = i

    if aux_index == -1:
        inicializado = False
    else:
        particulas[aux_index][5] = 1
        pos_actual = np.array([particulas[aux_index][0], particulas[aux_index][1]])

    return particulas, pos_actual


def seleccion (particulas):
    nuevas_particulas = []
    aleatorios = np.random.random(size=len(particulas))
    peso_acumulado = np.cumsum(particulas, axis=0)[:, 4]
    maximo = np.amax(peso_acumulado)
    if maximo != 0:
        total_peso = peso_acumulado / maximo
        for i in range(len(aleatorios)):
            aux = np.where(total_peso >= aleatorios[i])[0][0]
            nuevas_particulas.append(particulas[aux].copy())
    return nuevas_particulas

def difusion (particulas, esp):
    for p in particulas:
        ruido = np.random.randint(-esp, esp, size=2)
        p[0] = p[0]+ruido[0]
        p[1] = p[1]+ruido[1]
    return particulas

def prediccion(particulas, ultima_deteccion, pos_actual):
    if np.all(pos_actual != -1):
        if np.all(ultima_deteccion != -1):
            despl = pos_actual - ultima_deteccion
            for p in particulas:
                p[0] = p[0]+despl[0]
                p[1] = p[1]+despl[1]

    ultima_deteccion = pos_actual.copy()

    return particulas, ultima_deteccion, pos_actual


def pintar_particulas(frame, particulas, pred):
    for p in particulas:
        p1 = (p[0], p[1])
        p2 = (p[0]+p[3], p[1]+p[2])
        if p[5] == 1:
            if pred == 2:
                cv2.rectangle(frame, p1, p2, (255, 0, 255), 2)
                p[5] = 0
            else:
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                p[5] = 0
        else:
            if pred == 1:
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            elif pred == 0:
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, p1, p2, (255, 0, 255), 2)

    return frame


inicializado = False
min_HSV = np.array([0, 130, 230])
max_HSV = np.array([40, 255, 255])

ultima_deteccion = np.array([-1, -1])

URL_imagenes = "./dataParticulas/"
data = os.listdir(URL_imagenes)
data.sort()

for img in data:
    pos_actual = np.array([-1, -1])

    frame = cv2.imread(os.path.join(URL_imagenes, img))

    if not inicializado:
        particulas = inicializar_particulas(30, 30, 10, frame.shape[1], frame.shape[0])
        inicializado = True

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_filtrado = cv2.inRange(hsv_frame, min_HSV, max_HSV)
    particulas = evaluacion(frame_filtrado, particulas)

    particulas, pos_actual = estimacion(particulas, pos_actual)

    frame_particulas = pintar_particulas(frame.copy(), particulas, 0)
    cv2.imshow("Particulas", frame_particulas)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    particulas = seleccion(particulas)
    if len(particulas) > 0:
        frame_particulas_sel = pintar_particulas(frame.copy(), particulas, 1)
        cv2.imshow("Particulas", frame_particulas_sel)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        particulas = difusion(particulas, 40)
        frame_particulas_dif = pintar_particulas(frame.copy(), particulas, 0)
        cv2.imshow("Particulas", frame_particulas_dif)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        particulas, ultima_deteccion, pos_actual = prediccion(particulas, ultima_deteccion, pos_actual)
        frame_particulas_pred = pintar_particulas(frame.copy(), particulas, 2)
        cv2.imshow("Particulas", frame_particulas_pred)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
