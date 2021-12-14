# ENSAMBLADO DE MODELOS DE MÁQUINAS DE APRENDIZAJE PARA LA DETECCIÓN DE NOTICIAS FALSAS

Manual de usuario para proyecto de título de Andres Zapata "ENSAMBLADO DE MODELOS DE MÁQUINAS DE APRENDIZAJE PARA LA DETECCIÓN DE NOTICIAS FALSAS".

Este manual de usuario se encuentra disponible en el repositorio https://github.com/andres-zapata/deteccionDeRumoresConEnsamble.

En este repositorio se presenta la implementación del trabajo, en donde se tiene un backend basado en Tensorflow y Flask para el rápido entrenamiento de máquinas de aprendizaje y ensambles, y también se tiene una interfaz (realizada aparte del alcance del proyecto de título) para facilitar la comodidad de otros usuarios que quieran utilizar este proyecto.

Este proyecto se puede ejecutar de 2 formas, las cuales son:
- Instalación y ejecución local.
- Ejecución a través de Docker.

## Requerimientos
Se requiere tener instalado:
- Python 3.6 o 3.7.
- NodeJs.
- Angular.

En el caso de ocupar Docker, se requiere que se descargue e instale antes de ejecutar cualquier paso. Este programa se puede encontrar en https://www.docker.com/get-started.

## Descarga de Dataset

Para el entrenamiento de las maquinas de aprendizaje y los ensambles se requiere que se descargue el dataaset "Twitter16" que se puede encontrar en la siguiente direccion: https://drive.google.com/file/d/1o5Nx0wGNQcqEDz1jiiB9W3dbQyaeE_hG/view?usp=sharing. Una vez ya descargado se debe descomprimir y que los contenidos y copiarlos dentro de la carpeta "backend" de este proyecto.
## Instalación de forma Local

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
No es necesario tener descargado el dataset para desplegar el frontend, tan solo desde la consola de comando ir a la carpeta "interfaz" e instalar los módulos de node:
```bash
cd interfaz
npm install
```
Una vez instalados los requerimientos ya se puede ejecutar el frontend:
```bash
npm start
```

Debería cargar una página web en http://localhost:4200/. Esta página web funciona como una interfaz para facilitar el uso del backend.


## Instalacion y ejecucion con Docker

Para la ejecución del proyecto con Docker se debe primero tener instalado el programa Docker, una vez instalado y tenerlo corriendo, desde la consola de comandos se debe ir a la carpeta "backend" y construir la imagen:
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
Esto ya debería dejar corriendo el backend y el frontend los cuales se pueden acceder desde http://localhost:5000/ y http://localhost:4200/ respectivamente.

## Manejo de Interfaz

La interfaz debería verse como:

<img src='imgs/edges2cats.jpg' width="400px"/>

En donde se pueden entrenar las máquinas de aprendizaje y ensambles de forma secuencial.

En la parte superior de la página, si antes no se a cargado los tweets, se debe elegir el grupo de preprocesamiento que se quiera aplicar y apretar el botón "cargar" para que se cargue el dataset en el backend y se cree el modelo Word2Vec con el preprocesamiento elegido.

Una vez hecho esto se puede pasar a entrenar los máquinas de aprendizaje en la sección siguiente, en donde se deben elegir los parámetros y las máquinas de aprendizaje que se quiera sean parte del ensamble final, luego presionar el botón "entrenar" y esperar a que se carguen las imagenes a la derecha de la sección que indicara que se han entrenado las máquinas de aprendizaje.

Finalmente, se puede entrenar el ensamble de la misma forma, yendo a la última sección, eligiendo la técnica de ensamble que se desea utilizar y presionar el botón "entrenar" y esperar a que se carguen las imagenes a la derecha de la sección que indicara que se ha entrenado el ensamble.

## Precauciones

Se debe tener en cuenta que los pasos deben seguirse en secuencia para que el ensamble sea entrenado correctamente. También tener en cuenta que al ejecutar el backend con docker puede que se demore unos minutos.
