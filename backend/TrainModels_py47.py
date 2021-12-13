import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

import numpy as np
import os
import pickle
from modelos import *


# ### Problema: Determinar la veracidad de tweets, utilizando clasificación
# - Implementar un clasificador de post de tweets para validar su veracidad.
# - Para esto se utilizará redes neuronales y el árbol de  propagación de los tweets.

# # Parte 1: Procesamiento de datos y funcionalidades
# ## Parte 1.1: Cargar tweets para generar datos de entrenamiento
# ## Parte 1.3: Modelo

### tensorflow OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ## Parte 1.5: Entrenamiento y Pruebas
epochs = 200

# ## Parte 2: Modelo LSTM
# ### Modelo word2vec específico, LSTM
# ### Creando los vectores
# Se utilizan todos los tweets etiquetados como el universo de documentos, para crear el modelo de embedding w2v

from preprocesamiento import *

def validationKfold():
    f = open('preProcDataset_sub_sub.pckl', 'rb')
    X_train, X_test, Y_train, Y_test, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()    # #### K-Folds w2v50, LSTM


    result_score_w2v50 = []
    result_acc_w2v50   = []
    resultTime_w2v50   = []

    epochs = 200
    print('LSTM')

    result_score_w2v50_temp, result_acc_w2v50_temp, resultTime_w2v50_temp =  perform_kfold_validation_LSTM(
        X=X_train
        , Y=Y_train
        , _emb_size = w2v50_emb_size
        , k=5
        , _epochs = epochs,
        tree_max_num_seq=tree_max_num_seq
    )

    result_score_w2v50.append(result_score_w2v50_temp)
    result_acc_w2v50.append(result_acc_w2v50_temp)
    resultTime_w2v50.append(resultTime_w2v50_temp)

    print('LSTM 2')

    result_score_w2v50_temp, result_acc_w2v50_temp, resultTime_w2v50_temp =  perform_kfold_validation_LSTM2(
        X=X_train
        , Y=Y_train
        , _emb_size = w2v50_emb_size
        , w2v50_emb_size = w2v50_emb_size
        , k=5
        , _epochs = epochs,
        tree_max_num_seq=tree_max_num_seq
    )

    result_score_w2v50.append(result_score_w2v50_temp)
    result_acc_w2v50.append(result_acc_w2v50_temp)
    resultTime_w2v50.append(resultTime_w2v50_temp)

    print('RCNN')

    result_score_w2v50_temp, result_acc_w2v50_temp, resultTime_w2v50_temp =  perform_kfold_validation_RCNN(
        X=X_train
        , Y=Y_train
        , _emb_size = w2v50_emb_size
        , w2v50_emb_size = w2v50_emb_size
        , k=5
        , _epochs = epochs,
        tree_max_num_seq=tree_max_num_seq
    )

    result_score_w2v50.append(result_score_w2v50_temp)
    result_acc_w2v50.append(result_acc_w2v50_temp)
    resultTime_w2v50.append(resultTime_w2v50_temp)

    print('BIGRU')

    result_score_w2v50_temp, result_acc_w2v50_temp, resultTime_w2v50_temp =  perform_kfold_validation_BIGRU(
        X=X_train
        , Y=Y_train
        , _emb_size = w2v50_emb_size
        , w2v50_emb_size = w2v50_emb_size
        , k=5
        , _epochs = epochs,
        tree_max_num_seq=tree_max_num_seq
    )

    result_score_w2v50.append(result_score_w2v50_temp)
    result_acc_w2v50.append(result_acc_w2v50_temp)
    resultTime_w2v50.append(resultTime_w2v50_temp)


    # #### Parte  Gráficos: K-Fold w2v50, LSTM
    from graficos import graphic_data
    graphic_data(
        name = 'pre2_kfolds'
        , dataX   = [0,1,2,3,4]
        , dataY   = result_acc_w2v50
        , labels  = []
        , title   = 'Acc vs kFold'
        , x_label = 'Folds'
        , y_label = 'Acc'
        , x_min   = 0
        , y_min   = 0
        , x_max   = 4
        , y_max   = 1
    )

def trainModels(data):
    print("data", data)
    f = open('preProcDataset_sub_sub.pckl', 'rb')
    X_train, X_test, Y_train, Y_test, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()   

    data_final_acc_w2v50 = []
    data_final_loss_w2v50 = []
    epochs = data['epochs']

    if('LSTM' in data['maquinas']):
        history_final_w2v50_LSTM = []
        models_w2v50_LSTM = []

        history_final_w2v50_LSTM,models_w2v50_LSTM = perform_final_validation_LSTM(
            X=X_train
            , Y=Y_train
            , X_validate=X_test
            , Y_validate=Y_test
            , _emb_size=w2v50_emb_size
            , _epochs=data['epochs']
            , _dropout = data['dropout']
            , _unidadesDimensionales = data['unidadesDimensionales']
            , _funcionActivacion = data['funcionActivacion']
            , tree_max_num_seq=tree_max_num_seq
        )
        data_final_acc_w2v50.append(history_final_w2v50_LSTM[0].history['acc'])
        data_final_loss_w2v50.append(history_final_w2v50_LSTM[0].history['loss'])

    if('LSTM-2' in data['maquinas']):
        history_final_w2v50_LSTM2 = []
        models_w2v50_LSTM2 = []

        history_final_w2v50_LSTM2 ,models_w2v50_LSTM2 = perform_final_validation_LSTM2(
            X=X_train
            , Y=Y_train
            , X_validate=X_test
            , Y_validate=Y_test
            , _emb_size=w2v50_emb_size
            , _epochs=data['epochs']
            , _dropout = data['dropout']
            , _unidadesDimensionales = data['unidadesDimensionales']
            , _funcionActivacion = data['funcionActivacion']
            , tree_max_num_seq=tree_max_num_seq
        )
        data_final_acc_w2v50.append(history_final_w2v50_LSTM2[0].history['acc'])
        data_final_loss_w2v50.append(history_final_w2v50_LSTM2[0].history['loss'])

    if('RCNN' in data['maquinas']):
        history_final_w2v50_RCNN = []
        models_w2v50_RCNN = []

        history_final_w2v50_RCNN ,models_w2v50_RCNN = perform_final_validation_RCNN(
            X=X_train
            , Y=Y_train
            , X_validate=X_test
            , Y_validate=Y_test
            , _emb_size=w2v50_emb_size
            , _epochs=data['epochs']
            , _dropout = data['dropout']
            , _unidadesDimensionales = data['unidadesDimensionales']
            , _funcionActivacion = data['funcionActivacion']
            , tree_max_num_seq=tree_max_num_seq
        )
        data_final_acc_w2v50.append(history_final_w2v50_RCNN[0].history['acc'])
        data_final_loss_w2v50.append(history_final_w2v50_RCNN[0].history['loss'])

    if('BIGRU' in data['maquinas']):
        history_final_w2v50_bigru = []
        models_w2v50_bigru = []

        history_final_w2v50_bigru, models_w2v50_bigru = perform_final_validation_bigru(
            X=X_train
            , Y=Y_train
            , X_validate=X_test
            , Y_validate=Y_test
            , _emb_size=w2v50_emb_size
            , _epochs=data['epochs']
            , _dropout = data['dropout']
            , _unidadesDimensionales = data['unidadesDimensionales']
            , _funcionActivacion = data['funcionActivacion']
            , tree_max_num_seq=tree_max_num_seq
        )
        data_final_acc_w2v50.append(history_final_w2v50_bigru[0].history['acc'])
        data_final_loss_w2v50.append(history_final_w2v50_bigru[0].history['loss'])

    if('BIGRU-2' in data['maquinas']):
        history_final_w2v50_bigru2 = []
        models_w2v50_bigru2 = []

        history_final_w2v50_bigru2, models_w2v50_bigru2 = perform_final_validation_bigru2(
            X=X_train
            , Y=Y_train
            , X_validate=X_test
            , Y_validate=Y_test
            , _emb_size=w2v50_emb_size
            , tree_max_num_seq = tree_max_num_seq
            , _epochs=data['epochs']
            , _dropout = data['dropout']
            , _unidadesDimensionales = data['unidadesDimensionales']
            , _funcionActivacion = data['funcionActivacion']
        )
        data_final_acc_w2v50.append(history_final_w2v50_bigru2[0].history['acc'])
        data_final_loss_w2v50.append(history_final_w2v50_bigru2[0].history['loss'])

    if('Naive Bayes' in data['maquinas']):
        trainNaiveBayes(X_train, Y_train)

    if('SVM' in data['maquinas']):
        trainSVM(X_train, Y_train)

    if('Random Forest' in data['maquinas']):
        trainRandomForest(X_train, Y_train)

    print("Graficando")
    ##Gráfico de Accuracy
    from graficos import graphic_data

    imageNameAcc = graphic_data(
        name = 'pre2_acc'
        , dataX=np.arange(0,epochs,20)
        , dataY=data_final_acc_w2v50
        , labels=['Modelo 1','Modelo 2','Modelo 3','Modelo 4','Modelo 5']
        , title='Acc vs Epochs'
        , x_label= 'Epochs'
        , y_label = 'Acc'
        , x_min = 0
        , y_min = 0
        , x_max = epochs
        , y_max = 1+0.1
    )

    ##Gráfico de Loss 
    max_loss_w2v50 = 0
    for i in np.arange(0,len(data_final_loss_w2v50)):
        if(max(data_final_loss_w2v50[i]) > max_loss_w2v50):
            max_loss_w2v50 = max(data_final_loss_w2v50[i])

    imageNameLoss= graphic_data(
        name = 'pre2_loss'
        , dataX = np.arange(0,epochs,20)
        , dataY = data_final_loss_w2v50
        , labels=['Modelo 1','Modelo 2','Modelo 3','Modelo 4','Modelo 5']
        , title = 'Loss vs Epochs'
        , x_label = 'Epochs'
        , y_label = 'Loss'
        , x_min =  0
        , y_min = 0
        , x_max = epochs
        , y_max =  max_loss_w2v50+0.2
    )

    return imageNameAcc, imageNameLoss

     
