# Reconocedor de objetos

Primeramente deben ser instalados los siguiente modulos de python.

* pilow
* opencv
* numpy
* matplotlib
* skimage

El programa comienza leyendo una foto de la cual, a partir de algoritmos de machine learning, siendo especificos el algoritmo llamado: "montaña" calcula que region de pixeles pertenece a cada objeto de la foto, de tal manera que podemos saber cuantos objetos hay, y de que tamaño son estos.

El programa lleva a cabo multiples etapas para cumplir con su objetivo, entre estas se encuentra:

1. Foto ingresada pero rescalada al tamaño VGA.
1. Foto binarizada.
1. Foto binarizada con los clusters que se proponen inicialmente.
1. Foto binarizada con los clusters en una posicion final por haber utilizado el algoritmo de montaña.
1. Foto con multiples regiones de pixeles (cada region mostrado en diferente color y cada una de estas, perteneciente a un objeto en particular).

Una vez que el programa termina de hacer dichas etapas, muestra un menu del cual el usuario debe elegir una region de pixeles para que dicha region se pase a los colores originales y asi revelen un objeto seleccionado.