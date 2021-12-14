# ENSAMBLADO DE MODELOS DE MAQUINAS DE APRENDIZAJE PARA LA DETECCION DE NOTICIAS FALSAS

Manual de usuario para proyecto de titulo de Andres Zapata "ENSAMBLADO DE MODELOS DE MAQUINAS DE APRENDIZAJE PARA LA DETECCION DE NOTICIAS FALSAS".

Este manual de usuario se encuentra disponible en el repositorio https://github.com/andres-zapata/deteccionDeRumoresConEnsamble.

En este repositorio se presenta la implementacion del trabajo, en donde se tiene un backend basado en Tensorflow y Flask para el rapido entrenamiento de máquinas de aprendizaje y ensambles, y tambien se tiene una interfaz (realizada aparte del alcance del proyecto de titulo) para facilitar la comodidad de otros usuarios que quieran utilizar este proyecto.

Este proyecto se puede ejecutar de 2 formas, las cuales son:
- Instalacion y ejecucion local.
- Ejecucion a traves de Docker.

## Requerimientos
Se requiere tener instalado:
- Python 3.6 o 3.7.
- NodeJs.
- Angular.

En el caso de ocupar Docker, se requiere que se descargue e instale antes de ejecutar cualquier paso. Este programa se puede encontrar en https://www.docker.com/get-started.

## Descarga de Dataset

Para el entrenamiento de las maquinas de aprendizaje y los ensambles se requiere que se descargue el dataaset "Twitter16" que se puede encontrar en la siguiente direccion: https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip. Una vez ya descargado se debe descomprimir y que los contenidos de la "twitter16" sean copiados dentro de la carpeta "backend" de este proyecto. Luego se debe cambiar el nombre de la carpeta "source_tweets.txt" por "post.txt"

## Instalacion de forma Local

### Preparacion ejecucion de backend
Una vez descargado este repositorio y el desplegado el dataset, desde la consola de comando ir a la carpeta "backend" e instalar los requerimientos:
```bash
cd backend
pip install -r requirements.txt
```

Una vez instalados los requerimientos ya se puede ejecutar el backend:
```bash
python main.py
```
Una vez corriendo el backend se pueden hacer llamadas post a las rutas indicadas en main.py para entrenar cargar el dataset, entrenar los modelos y entrenar los ensambles.

### Preparacion ejecucion de frontend
No es necesario tener descargado el dataset para desplegar el frontend, tan solo desde la consola de comando ir a la carpeta "interfaz" e instalar los modulos de node:
```bash
cd interfaz
npm install
```
Una vez instalados los requerimientos ya se puede ejecutar el frontend:
```bash
npm start
```

Lo que deberia cargar una pagina web en http://localhost:4200/. Esta pagina web funciona como una interfaz para facilitar el uso del backend.


## Instalacion y ejecucion con Docker

Para la ejecucion del proyecto con Docker se debe primero tener instalado el programa Docker, una vez instalado y tenerlo corriendo, desde la consola de comandos se debe ir a la carpeta "backend" y construir la imagen:
```bash
cd backend
docker build -t backend .
```
Una vez construida la imagen se puede ejecutar el contenedor:
```bash
docker run -p 5000:5000 backend
```

Para ejecutar el frontend son los mismos pasos. Desde la consola de comandos se debe ir a la carpeta "interfaz" y construir la imagen:
```bash
cd interfaz
docker build -t interfaz .
```
Una vez construida la imagen se puede ejecutar el contenedor:
```bash
docker run -p 4200:4200 interfaz
```
Esto ya deberia dejar corriendo el backend y el frontend los cuales se pueden acceder desde http://localhost:5000/ y http://localhost:4200/ respectivamente.

## Manejo de Interfaz

La interfaz deberia verse como:

<img src='imgs/edges2cats.jpg' width="400px"/>

En donde se pueden entrenar las máquinas de aprendizaje y ensambles de forma secuencial.

En la parte superior de la pagina, si antes no se a cargado los tweets, se debe elegir el grupo de preprocesamiento que se quiera aplicar y apretar el boton "cargar" para que se cargue el dataset en el backend y se cree el modelo Word2Vec con el preprocesamiento elegido.

Una vez echo esto se puede pasar a entrenar los modelos en la seccion siguiente, en donde 
