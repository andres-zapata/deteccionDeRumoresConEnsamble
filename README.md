# ENSAMBLADO DE MODELOS DE MAQUINAS DE APRENDIZAJE PARA LA DETECCION DE NOTICIAS FALSAS

Manual de usuario para proyecto de titulo de Andres Zapata "ENSAMBLADO DE MODELOS DE MAQUINAS DE APRENDIZAJE PARA LA DETECCION DE NOTICIAS FALSAS".

Este manual de usuario se encuentra disponible en el repositorio https://github.com/andres-zapata/deteccionDeRumoresConEnsamble.

En este repositorio se presenta la implementacion del trabajo, en donde se tiene un backend basado en Tensorflow y Flask para el rapido entrenamiento de máquinas de aprendizaje y ensambles, y tambien se tiene una interfaz (realizada aparte del alcance del proyecto de titulo) para facilitar la comodidad de otros usuarios que quieran utilizar este proyecto.

Este proyecto se puede ejecutar de 2 formas, las cuales son:
- Instalacion y ejecucion local.
- Ejecucion a traves de Dockers.

## Requerimientos
Se requiere tener instalado:
- Python 3.6 o 3.7.
- NodeJs.
- Angular.

En el caso de ocupar Dockers, se requiere que se descargue e instale antes de ejecutar cualquier paso. Este programa se puede encontrar en https://www.docker.com/get-started.

## Descarga de Dataset

Para el entrenamiento de las maquinas de aprendizaje y los ensambles se requiere que se descargue el dataaset "Twitter16" que se puede encontrar en la siguiente direccion: https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip. Una vez ya descargado se debe descomprimir y que los contenidos de la "twitter16" sean copiados dentro de la carpeta "backend" de este proyecto. Luego se debe cambiar el nombre de la carpeta "source_tweets.txt" por "post.txt"

## Instalacion de forma Local

### Preparacion backend
Una vez descargado este repositorio y el desplegado el dataset, desde la consola de comando ir a la carpeta "backend" e instalar los requerimientos:
```bash
cd backend
pip install -r requirements.txt
```
