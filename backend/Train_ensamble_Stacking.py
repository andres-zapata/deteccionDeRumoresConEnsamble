import tensorflow.compat.v1 as tf
from preprocesamiento import *

def trainEnsambleStacking(data):
    tf.disable_v2_behavior()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    import numpy as np
    import pickle

    # Esto carga la variable con los tweets ya parseados,
    # si se quiere parsear otro dataset ocupar script CargarTweets.py

    f = open('parsedTweets.pckl', 'rb')
    seqs_lens, labeled_posts, all_posts, number_of_tweets = pickle.load(f)
    f.close()


    # Categorias: true, false, unverified, non-rumor
    categories = ['true', 'false', 'unverified', 'non-rumor']
    num_categories = len(categories)

    # build vocabulary and train model
    w2v50_emb_size = 50
    epochs = 200

    #"sub" se refiere a "subdivision"
    f = open('preProcDataset_sub.pckl', 'rb')
    X_train_sub, X_test_sub, Y_train_sub, Y_test_sub, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()   

    f = open('preProcDataset_sub_sub.pckl', 'rb')
    X_train_sub_sub, X_test_sub_sub, Y_train_sub_sub, Y_test_sub_sub, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()   


    models_w2v50 = []

    if('LSTM' in data['maquinas']):
        modeloLSTM  = tf.keras.models.load_model('model_LSTM.h5')
        models_w2v50.append(modeloLSTM)

    if('LSTM-2' in data['maquinas']):
        modeloLSTM2 = tf.keras.models.load_model('model_LSTM2.h5')
        models_w2v50.append(modeloLSTM2)

    if('RCNN' in data['maquinas']):
        modeloRCNN  = tf.keras.models.load_model('model_RCNN.h5')
        models_w2v50.append(modeloRCNN)

    if('BIGRU' in data['maquinas']):
        modeloBIGRU = tf.keras.models.load_model('modelBIGRU.h5')
        models_w2v50.append(modeloBIGRU)

    if('BIGRU-2' in data['maquinas']):
        modeloBIGRU2= tf.keras.models.load_model('modelBIGRU2.h5')
        models_w2v50.append(modeloBIGRU2)

    from joblib import load
    modelsClassic = []
    if('Naive Bayes' in data['maquinas']):
        modeloNB = load('naiveBayesModel.joblib')
        models_w2v50.append(modeloNB)
    
    if('SVM' in data['maquinas']):
        modeloSVM = load('svmModel.joblib')
        models_w2v50.append(modeloSVM)
    
    if('Random Forest' in data['maquinas']):
        modeloRF = load('randomForestModel.joblib')
        models_w2v50.append(modeloRF)


    from ensambles import fit_stacked_model

    Y_test_sub_sub_labeled=np.argmax(Y_test_sub_sub,1)
    history_ensemble = []

    X_test_sub_2d      = X_test_sub.reshape((X_test_sub.shape[0],X_test_sub.shape[1]*X_test_sub.shape[2]))
    Y_test_sub_labeled = np.argmax(Y_test_sub_sub,1)

    print("models_w2v50", models_w2v50)
    ensamble, history_ensemble = fit_stacked_model(
        models = models_w2v50,
        modelsClassic = modelsClassic,
        inputX = X_test_sub_sub, 
        inputXClassic= X_test_sub_2d, 
        inputY = Y_test_sub_sub_labeled, 
        _epochs = epochs, 
        num_categories=num_categories
    )

    from graficos import graphic_data

    imageEnsambleAcc = "pre2_stacking_acc"
    imageEnsambleLoss = "pre2_stacking_loss"
    imageEnsambleMatrix = "pre2_stacking_confusion_matrix"
    
    #Graficos Acuraccy Ensamble Stacking
    graphic_data(
        name=imageEnsambleAcc,
        dataX=np.arange(0, epochs, 20), 
        dataY=[history_ensemble.history['acc']],
        labels=["ensamble"], 
        title="Acc vs Epochs, Ensamble Stacking",
        x_label="Epochs", 
        y_label="%", 
        x_min=0, 
        y_min=0, 
        x_max=epochs, 
        y_max=1+0.1
    )

    #Graficos Loss Ensamble Stacking
    graphic_data(
        name=imageEnsambleLoss,
        dataX=np.arange(0, epochs, 20), 
        dataY=[history_ensemble.history['loss']],
        labels=['ensamble'], 
        title="Loss vs epochs, Ensamble Stacking", 
        x_label="Epochs", 
        y_label="Loss", 
        x_min=0, 
        y_min=0, 
        x_max=epochs, 
        y_max=max(history_ensemble.history['loss'])+0.2
    )

    from ensambles import stacked_prediction
    from graficos import plot_conf_matrix

    # evaluate model on test set
    yhat = stacked_prediction(models_w2v50, ensamble, X_test_sub)

    yhatArgmax=[np.argmax(prediccion) for prediccion in yhat]
    Y_test_subArgmax = [np.argmax(truth) for truth in Y_test_sub]

    Y_true_categorical = tf.keras.utils.to_categorical(Y_test_subArgmax, num_classes=4)
    Y_pred_categorical = tf.keras.utils.to_categorical(yhatArgmax, num_classes=4)

    plot_conf_matrix(
        y_true= Y_true_categorical, 
        y_pred= Y_pred_categorical, 
        _title=imageEnsambleMatrix
    )

    from sklearn import metrics
    dictionary = dict(zip(np.arange(4), categories))
    print("muestras: ", len(X_test_sub))
    print(dictionary)
    print(metrics.classification_report(yhatArgmax, Y_test_subArgmax, digits=3))

    return imageEnsambleAcc, imageEnsambleLoss, imageEnsambleMatrix



