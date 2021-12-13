import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from CargarTweets import requestCargarTweets
from TrainModels_py47 import trainModels
from Train_ensamble_Stacking import trainEnsambleStacking
from train_ensamble_promedio import trainEnsamblePromedio
from train_ensamble_bagging import trainEnsambleBagging
from crearModeloWord2Vec import crearModeloWord2Vec
from flask import send_file

app = Flask(__name__)
CORS(app)

@app.route('/')
def test():
    return jsonify(
    message="ok",
    category="success",
    status=200
)

@app.route('/cargarTweets', methods = ['POST'])
def endpointCargarTweets():
    requestCargarTweets()
    return jsonify(
        message="ok",
        category="success",
        status=200
    )

@app.route('/trainModels', methods = ['POST'])
def endpointTrainModels():
    data = request.json
    imageNameAcc, imageNameLoss = trainModels(data)
    return {"accRoute": imageNameAcc, "lossRoute" : imageNameLoss}

@app.route('/trainEnsambleStacking', methods = ['POST'])
def endpointTrainEnsambleStacking():
    print('Entrenando ensamble stacking...')
    data = request.json
    imageEnsambleAcc, imageEnsambleLoss, imageEnsambleMatrix = trainEnsambleStacking(data)
    return {
        "accRoute"   : imageEnsambleAcc+'.png', 
        "lossRoute"  : imageEnsambleLoss+'.png',
        "matrixRoute": imageEnsambleMatrix+'.png',
    }

@app.route('/trainEnsamblePromedio', methods = ['POST'])
def endpointTrainEnsamblePromedio():
    print('Entrenando ensamble promedio...')
    data = request.json
    imageEnsambleMatrix= trainEnsamblePromedio(data)
    return {
        "matrixRoute": imageEnsambleMatrix+'.png'
    }

@app.route('/trainEnsambleBagging', methods = ['POST'])
def endpointTrainEnsambleBagging():
    print('Entrenando ensamble promedio...')
    data = request.json
    imageEnsambleAcc, imageEnsambleLoss, imageEnsambleMatrix, metricas = trainEnsambleBagging(data)
    return {
        "accRoute"   : imageEnsambleAcc+'.png', 
        "lossRoute"  : imageEnsambleLoss+'.png',
        "matrixRoute": imageEnsambleMatrix+'.png',
        "metricas"   : metricas
    }

@app.route('/generarDataset', methods = ['POST'])
def endpointcrearModeloWord2Vec():
    data = request.json
    crearModeloWord2Vec(data)
    return jsonify(
        message="ok",
        category="success",
        status=200
    )

@app.route('/postImage', methods = ['POST'])
def endpointpostImage():
    data = request.json
    return send_file(data['url'], mimetype='image/gif')

if __name__ == '__main__':
	app.run(host="0.0.0.0")