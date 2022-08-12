# No tomarle fotos a objetos que produzcan algun reflejo o al menos cuidar que en ese instante no 
# lo hagan.
# No tomarle fotos a objetos con huecos, como una cinta.
# Foto1*0 #Para que la foto reescalada quede completamante negra y de manera inversa si se multiplica
# por 255 queda la foto completamente blanca.
# Los objetos no deben estar tan juntos o al menos en la foto no verse tan juntos.
# Viendo unicamente la foto binarizada se sabra si sera revelado correctamente o no determinado objeto

from ssl import SSL_ERROR_SYSCALL
from PIL import Image
import cv2
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import math

# GRAFICACION DE OBJETO SELECCIONADO
def seleccionado(objeto, Foto1):
    mostrar = np.zeros(Foto1.shape)

    for  i in range(len(objeto[0, :])):
        for k in range(3):
            mostrar[int(objeto[1, i]), int(objeto[0, i]), k] = Foto1[int(objeto[1, i]), int(objeto[0, i]), k] 

    return mostrar


#LEER FOTO
# Foto = io.imread("2objetos.jpeg")
# Foto = io.imread("3objetos.jpeg")
Foto = io.imread("5objetos.jpeg")

#REESCALARAR FOTO
plt.figure()
Foto1 = cv2.resize(Foto, (640, 480))
plt.imshow(Foto1, vmin = 0, vmax = 1)

#GRISES
R, G, B = Foto1[:, :, 0], Foto1[:, :, 1], Foto1[:, :, 2]
grises = 0.2989 * R + 0.5870 * G + 0.1140 * B

#BINARIZACION
plt.figure()
binaria = grises > 150
plt.imshow(binaria, cmap = 'gray')


#INICIALIZACION DE DATOS/PIXELES
unos = np.where(binaria == True)
coord = np.zeros((2, len(unos[1])))
coord[0, :] = unos[1]
coord[1, :] = unos[0]

#MONTAÑASO 2D
n = len(coord[0, :])

nodos = np.array([np.arange(0, 774, 70.4),
                np.arange(0, 774, 70.4)]) #640 + 64 = 704 -> 704/10 = 70.4

print('nodos:')
print(nodos)

lnodos = len(nodos[0, :])

alpha = 0.009
betha = 0.009
delta = 10

montanas = np.zeros((lnodos, lnodos))
M = np.zeros((lnodos, lnodos))
suma = 0
for j in range(lnodos):
    for i in range(lnodos):
        for k in range(n):
            m = math.exp(-alpha*math.sqrt( (coord[0, k] - nodos[0, j])**2 + (coord[1, k] - nodos[1, i])**2) )
            M[i, j] = suma + m
            suma = M[i, j]
        suma = 0
M1 = M


p0 = np.zeros(lnodos)
p1 = np.zeros(lnodos)
C  = np.zeros(lnodos)

maxX = []
maxY = []
M2 = np.zeros((lnodos, lnodos))
Mactual = np.zeros((1, lnodos, lnodos))

#DICICONARIO DE VALORES PARA DELTA
# 2objetos: 5,2.5 3objetos: 5, 5objetos: 15;
k = 0
while(1):
    C[k] = M.max()

    print('k:', k)
    pos = np.where(M == C[k])
    p0[k], p1[k] = pos[0], pos[1]
    
    g = [0]*(k + 1)
    for j in range(lnodos):
        for i in range(lnodos):
            for z in range(k + 1):
                dc = math.sqrt( (nodos[0, int(p0[z])] - nodos[0, j])**2 + (nodos[1, int(p1[z])] - nodos[1, i])**2)
                g[z] = math.exp(-betha*dc)
                g = np.append(g, 0)
            mc = sum(g)
            M2[j, i] = M[j, i] - M[int(p0[k]), int(p1[k])]*mc
            M2 = np.maximum(M2, 0)
    
    Mactual[k, :, :] = M2

    k += 1
    c2 = M2.max()
    delta = C[0]/c2
    M = M2

    maxX.append(nodos[0, int(p0[z])])
    maxY.append(nodos[1, int(p1[z])])
    
    if delta > 15:
        break

    Mactual = np.append(Mactual, np.zeros((1, lnodos, lnodos)), axis = 0)


plt.figure('Conjunto de datos y Centros Iniciales', figsize = (6.4, 5.0))
plt.ylabel('y')
plt.xlabel('x')
plt.title('Distribucion y Centros')
for i in range(k):
    plt.plot(nodos[1, int(p1[i])], nodos[0, int(p0[i])], 'r*')
    print('Centro ', i + 1, ': (', 70.4*p1[i], ',', 70.4*p0[i], ')')
plt.scatter(coord[0, :], coord[1, :], color = 'k')

plt.xticks(np.arange(0, 774, 70.4))
plt.yticks(np.arange(0, 563, 70.4))
plt.grid()

# plt.show()

#K-MEANS JANG
#1
c = k
centros = np.zeros((2, c))

for i in range(c):
    centros[0, i] = 64 * p1[i]
    centros[1, i] = 64 * p0[i]
J_n = 0

#2 iteriativo 
d = np.zeros((c, n))
u_0 = d

while(1):
    for j in range(c):
        for i in range(n):
            d[j, i] = math.sqrt((coord[0, i] - centros[0, j])**2 + (coord[1, i] - centros[1, j])**2)

    u_1 = np.zeros((c, n))
    minimos = d.min(axis = 0)
    for i in range(n):
        for j in range(c):
            if(d[j, i] == minimos[i]):
                u_1[j, i] = 1

    #3
    J_1 = np.sum(d)
    print('La funcion de costo es:', J_1)

    if( J_1 != J_n ):
        print('Funcion costo distinta. Recalculando')
        J_n = J_1
    else:
        print('Funcion costo igual.')
        break

    #4 
    for j in range(c):
        x_datos = 0
        y_datos = 0
        for i in range(len(u_1[0, :])): 
            x_datos = x_datos + u_1[j, i]*coord[0, i]        
            y_datos = y_datos + u_1[j, i]*coord[1, i] 
            
        c_i = sum(u_1[j, :]) 
        centros[0, j] = x_datos/c_i
        centros[1, j] = y_datos/c_i

print('Los centros son: ')
for i in range(len(centros[0, :])):
    print('{', centros[0, i], ',', centros[1, i], '}')


fig = plt.figure('Jang Conjunto de Puntos y Centros Finales', figsize = (6.4, 5.0))

plt.ylabel('y')
plt.xlabel('x')
plt.xlim([0, 640])
plt.ylim([0, 480])
plt.scatter(coord[0, :], coord[1, :], c = 'k')
plt.scatter(centros[0, :], centros[1, :], c = 'r', marker = '*')
plt.legend(['Muestras', 'Centros'])



#EXTRACCION DE OBJETOS
objeto1 = np.zeros((2, 1))
objeto2 = np.zeros((2, 1))
objeto3 = np.zeros((2, 1))
objeto4 = np.zeros((2, 1))
objeto5 = np.zeros((2, 1))

k1 = 0
k2 = 0
k3 = 0
k4 = 0
k5 = 0
for i in range(n):
    indice_pixel = np.argmax(u_1[:, i])

    if(indice_pixel == 0):
        objeto1[0, k1] = coord[0, i]
        objeto1[1, k1] = coord[1, i]
        k1 += 1
        objeto1 = np.append(objeto1, [[0], [0]],  1)

    elif(indice_pixel == 1):
        objeto2[0, k2] = coord[0, i]
        objeto2[1, k2] = coord[1, i]
        k2 += 1
        objeto2 = np.append(objeto2, [[0], [0]], 1)

    elif(indice_pixel == 2):
        objeto3[0, k3] = coord[0, i]
        objeto3[1, k3] = coord[1, i]
        k3 += 1
        objeto3 = np.append(objeto3, [[0], [0]],  1)

    elif(indice_pixel == 3):
        objeto4[0, k4] = coord[0, i]
        objeto4[1, k4] = coord[1, i]
        k4 += 1
        objeto4 = np.append(objeto4, [[0], [0]],  1)

    else:
        objeto5[0, k5] = coord[0, i]
        objeto5[1, k5] = coord[1, i]
        k5 += 1
        objeto5 = np.append(objeto5, [[0], [0]],  1)


#GRAFICACION DE OBJETOS 
fig = plt.figure('Objetos', figsize = (6.4, 5.0))

plt.title('Objetos')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim([0, 640])
plt.ylim([0, 480])
plt.scatter(objeto1[0, :-1], objeto1[1, :-1], color = '#870000')
plt.scatter(objeto2[0, :-1], objeto2[1, :-1], color = '#182848')
plt.scatter(objeto3[0, :-1], objeto3[1, :-1], color = '#480048')
plt.scatter(objeto4[0, :-1], objeto4[1, :-1], color = '#215f00') 
plt.scatter(objeto5[0, :-1], objeto5[1, :-1], color = '#F09819') 

plt.legend(['Vino', 'Azul', 'Purpura', 'Verde', 'Amarillo'])

fig.show()
#MENU
print('\n1. Vino\n2. Azul\n3. Purpura\n4. Verde\n5. Amarillo\n')
opc = int(input('Escriba el numero de objeto que desea visualizar: '))

# OBJETO SELECCIONADO
fig = plt.figure('Objeto Seleccionado', figsize = (6.4, 5.0))
 
if opc == 1:
    mostrar = seleccionado(objeto1, Foto1)

elif opc == 2:
    mostrar = seleccionado(objeto2, Foto1)

elif opc == 3:
    mostrar = seleccionado(objeto3, Foto1)

elif opc == 4:
    mostrar = seleccionado(objeto4, Foto1)

else:
    mostrar = seleccionado(objeto5, Foto1)  

plt.imshow(np.uint8(mostrar), vmin = 0, vmax = 1)

plt.show()