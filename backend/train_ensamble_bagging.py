import tensorflow.compat.v1 as tf
import numpy as np
import os
from ensambles import creacionEnsambleBagging
from modelos import *


def trainEnsambleBagging(data):    
    tf.disable_v2_behavior()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    UNITS = 128
    DROPOUT = 0.5

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

    w2v50_emb_size = 50
    epochs = 200



    f = open('preProcDataset_sub', 'rb')
    X_train, X_test, Y_train, Y_test, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()   

    f = open('preProcDataset_sub_sub.pckl', 'rb')
    X_train_sub, X_test_sub, Y_train_sub, Y_test_sub, tree_max_num_seq, w2v50_emb_size = pickle.load(f)
    f.close()   


    ### tensorflow OMP: Error #15
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    epochs = 200
    models = []

    print("data ", data['maquinas'])

    if('LSTM' in data['maquinas']):
        print('LSTM-------------------------------------')
        model = create_model_LSTM(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , num_categories= num_categories
            , units=UNITS
            , _dropout=DROPOUT)
        models.append(model)

    if('LSTM-2' in data['maquinas']):
        print('LSTM2-------------------------------------')
        model = create_model_StackedLSTM(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= num_categories
            , _units=UNITS
            , _dropout=DROPOUT)
        models.append(model)

    if('RCNN' in data['maquinas']):
        print('RCNN-------------------------------------')
        model = create_model_BI_GRU(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= num_categories
            , _units=UNITS
            , _dropout=DROPOUT)
        models.append(model)

    if('BIGRU' in data['maquinas']):
        print('BIGRU-------------------------------------')
        model = create_model_RCNN(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= num_categories
            , _units=UNITS
            , _dropout=DROPOUT)
        models.append(model)

    if('BIGRU-2' in data['maquinas']):
        print('BIGRU2-------------------------------------')
        model = create_model_BI_StackedGRU(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= num_categories
            , _units=UNITS
            , _dropout=DROPOUT)
        models.append(model)

    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    

    modelsClassic = []
    if('Naive Bayes' in data['maquinas']):
        modelsClassic.append(GaussianNB())
    
    if('SVM' in data['maquinas']):
        modelsClassic.append(SVC(kernel ='linear', C = 1))
    
    if('Random Forest' in data['maquinas']):
        modelsClassic.append(RandomForestClassifier(n_estimators=100))

    if( models != [] ):
        print("models ", models)
        modelos, trainingLog =  creacionEnsambleBagging(
            modelos=models
            , X_train=X_train_sub
            , Y_train=Y_train_sub
        )

        from graficos import graphic_data
        imageEnsambleAcc = "pre2_bagging_acc"
        imageEnsambleLoss = "pre2_bagging_loss"
        imageEnsambleMatrix = "pre2_bagging_confusion_matrix"

        #Preparando los datos para graficar
        data_final_acc_w2v50 = []
        data_final_loss_w2v50 = []

        for i in np.arange(0,len(trainingLog)):
            data_final_acc_w2v50.append(trainingLog[i].history['acc'])


        for i in np.arange(0,len(trainingLog)):
            data_final_loss_w2v50.append(trainingLog[i].history['loss'])

        graphic_data(
            name =imageEnsambleAcc
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

        graphic_data(
            name = imageEnsambleLoss
            , dataX = np.arange(0,epochs,20)
            , dataY = data_final_loss_w2v50
            , labels=['LSTM','RCNN','LSTM2','BI-GRU','BI-GRU2']
            , title = 'Loss vs Epochs'
            , x_label = 'Epochs'
            , y_label = 'Loss'
            , x_min =  0
            , y_min = 0
            , x_max = epochs
            , y_max =  max_loss_w2v50+0.2
        )

    if(modelsClassic != []):
        print("modelsClassic", modelsClassic)
        from ensambles import creacionEnsambleBagging_clasico

        #reformar los dataset para ocupar en naive bayes
        X_sub_train_2d      = X_train_sub.reshape((X_train_sub.shape[0],X_train_sub.shape[1]*X_train_sub.shape[2]))
        Y_sub_train_labeled = np.argmax(Y_train_sub,1)

        X_sub_test_2d       = X_test_sub.reshape((X_test_sub.shape[0], X_test_sub.shape[1]*X_test_sub.shape[2]))
        Y_sub_test_labeled  = np.argmax(Y_test_sub,1)


        modelosClasicos =  creacionEnsambleBagging_clasico(
            modelos=modelsClassic
            , X_train=X_sub_train_2d
            , Y_train=Y_sub_train_labeled
        )


    print("testYClasicos", Y_sub_test_labeled)
    print("testY", Y_test_sub)


    from ensambles import promedioPredicciones
    predPromediadas = promedioPredicciones(
        modelos = modelos
        , modelosClasicos=modelosClasicos
        , dataset = X_test_sub
        , datasetClasicos= X_sub_test_2d
    )

    pr = np.array(predPromediadas)

    prediccionesArgmax=[np.argmax(prediccion) for prediccion in predPromediadas]
    verdadesArgmax = [np.argmax(truth) for truth in Y_test_sub]

    from graficos import plot_conf_matrix

    Y_pred_categorical = tf.keras.utils.to_categorical(prediccionesArgmax,num_classes=4, dtype="int")
    Y_true_categorical = tf.keras.utils.to_categorical(verdadesArgmax    ,num_classes=4, dtype="int")


    plot_conf_matrix(
        y_true=Y_true_categorical, 
        y_pred=Y_pred_categorical, 
        _title=imageEnsambleMatrix
    )

    from sklearn import metrics
    dictionary = dict(zip(np.arange(4), categories))
    print("muestras: ", len(X_test_sub))
    print(dictionary)
    metricas = metrics.classification_report(verdadesArgmax, prediccionesArgmax, digits=3)

    return imageEnsambleAcc, imageEnsambleLoss, imageEnsambleMatrix, metricas
