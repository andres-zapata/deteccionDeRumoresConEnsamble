
import tensorflow.compat.v1 as tf
import numpy as np
from ensambles import promedioPredicciones

def trainEnsamblePromedio(data):
    tf.disable_v2_behavior()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    import pickle


    # Categorias: true, false, unverified, non-rumor
    categories = ['true', 'false', 'unverified', 'non-rumor']
    num_categories = len(categories)

    #entrega un vector one-hot de la categoria, de largo 4 (por el número de categorias)
    def to_category_vector(category):
        vector = np.zeros(len(categories)).astype(np.float32)
        
        for i in range(len(categories)):
            if categories[i] == category:
                vector[i] = 1.0
                break
        
        return vector


    epochs = 200


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

    if('LSTM2' in data['maquinas']):
        modeloLSTM2 = tf.keras.models.load_model('model_LSTM2.h5')
        models_w2v50.append(modeloLSTM2)

    if('RCNN' in data['maquinas']):
        modeloRCNN  = tf.keras.models.load_model('model_RCNN.h5')
        models_w2v50.append(modeloRCNN)

    if('BIGRU' in data['maquinas']):
        modeloBIGRU = tf.keras.models.load_model('modelBIGRU.h5')
        models_w2v50.append(modeloBIGRU)

    if('BIGRU2' in data['maquinas']):
        modeloBIGRU2= tf.keras.models.load_model('modelBIGRU2.h5')
        models_w2v50.append(modeloBIGRU2)

    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    
    from joblib import dump, load
    modelsClassic = []
    if('Naive Bayes' in data['maquinas']):
        modeloNB = load('naiveBayesModel.joblib')
        modelsClassic.append(modeloNB)
    
    if('SVM' in data['maquinas']):
        modeloSVM = load('svmModel.joblib')
        modelsClassic.append(modeloSVM)
    
    if('Random Forest' in data['maquinas']):
        modeloRF = load('randomForestModel.joblib')
        modelsClassic.append(modeloRF)


    from ensambles import promedioPredicciones


    X_test_sub_2d      = X_test_sub.reshape((X_test_sub.shape[0],X_test_sub.shape[1]*X_test_sub.shape[2]))
    Y_test_sub_labeled = np.argmax(Y_test_sub,1)

    predPromediadas = promedioPredicciones(
        modelos = models_w2v50
        , modelosClasicos=modelsClassic
        , dataset = X_test_sub
        , datasetClasicos= X_test_sub_2d
    )   

    pr = np.array(predPromediadas)

    prediccionesArgmax=[np.argmax(prediccion) for prediccion in predPromediadas]
    verdadesArgmax = [np.argmax(truth) for truth in Y_test_sub]

    from graficos import plot_conf_matrix

    imageEnsambleMatrix = "pre2_stacking_confusion_matrix"

    Y_pred_categorical = tf.keras.utils.to_categorical(prediccionesArgmax,num_classes=4, dtype="int")
    Y_true_categorical = tf.keras.utils.to_categorical(verdadesArgmax    ,num_classes=4, dtype="int")


    plot_conf_matrix(
        y_true=Y_true_categorical, 
        y_pred=Y_pred_categorical, 
        _title=imageEnsambleMatrix
    )

    from sklearn.metrics import f1_score

    f1_scores = f1_score(
        y_true = Y_true_categorical
        , y_pred = Y_pred_categorical
        , average= None
    )

    from metricas import get_metricas

    acc, precision, recall, f1_score = get_metricas(prediccionesArgmax, verdadesArgmax)

    print("acc:", acc)
    print("precision:", precision)
    print("recall:", recall)
    #print("f1-score:", f1_score)

    print("true f1-score", f1_scores[0])
    print("false f1-score", f1_scores[1])
    print("unverified f1-score", f1_scores[2])
    print("non-rumor f1-score", f1_scores[3])

    from sklearn import metrics
    dictionary = dict(zip(np.arange(4), categories))
    print("muestras: ", len(X_test_sub))
    print(dictionary)
    print(metrics.classification_report(verdadesArgmax, prediccionesArgmax, digits=3))

    return imageEnsambleMatrix
