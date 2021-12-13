import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow.compat.v1 as tf
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from TweetModels import *

def creacionEnsambleBagging(modelos, X_train, Y_train):
    trainingLog = []
    n_splits = len(modelos)
    for _ in range(n_splits):
        # guarda los indices
        ix = [i for i in range(len(X_train))]
        train_ix = resample(ix, replace=True, n_samples=300)
        test_ix = [x for x in ix if x not in train_ix]

        # select data
        trainX, trainY = X_train[train_ix], Y_train[train_ix]
        testX, testY = X_train[test_ix], Y_train[test_ix]

        history = modelos[_].fit(trainX, trainY, batch_size=128, epochs=200, verbose=0, use_multiprocessing=False)
        trainingLog.append(history)

    return modelos, trainingLog

def creacionEnsambleBagging_clasico(modelos, X_train, Y_train):
    n_splits = len(modelos)
    for _ in range(n_splits):
        # guarda los indices
        ix = [i for i in range(len(X_train))]
        train_ix = resample(ix, replace=True, n_samples=300)
        test_ix = [x for x in ix if x not in train_ix]
        # select data
        trainX, trainY = X_train[train_ix], Y_train[train_ix]
        testX, testY = X_train[test_ix], Y_train[test_ix]

        modelos[_].fit(trainX, trainY)

    return modelos

def promedioPredicciones(modelos, modelosClasicos, dataset, datasetClasicos):
    predictions = []
    for model in modelos:
        predictions.append(model.predict(dataset))
    predictions = np.array(predictions)

    predictionsClasicos = []
    for model in modelosClasicos:
        predictionsClasicos.append(model.predict(datasetClasicos))
    predictionsClasicos = np.array(predictionsClasicos)

    if(predictionsClasicos.ndim == 2):
        predictionsClasicos=tf.keras.utils.to_categorical(predictionsClasicos, num_classes=4, dtype="int")

    promedios = []
    means = 0
    for j in range(len(predictions)):
        means = np.add(means,predictions[j])

    for j in range(len(predictionsClasicos)):
        means = np.add(means,predictionsClasicos[j])

    means = means/(len(predictions)+len(predictionsClasicos))
    promedios = means

    return promedios


def return_stacked_dataset(models, inputX, inputXClassic):
    from tensorflow.keras.utils import to_categorical
    stack_pred = None
    
    for i,model in enumerate(models):
        print(i)
        print("inputX", inputX.shape)

        # hacer predicciones
        try:
            predicciones = model.predict(inputX)
            prediccion=[np.argmax(prediccion) for prediccion in predicciones]
        except:
            prediccion = model.predict(inputXClassic)

        prediccion_oneHot = []
        for i in range(len(prediccion)):
            prediccion_oneHot.append(to_categorical(prediccion[i], num_classes=4))
        prediccion_oneHot = np.array(prediccion_oneHot)

        print("prediccion before", prediccion_oneHot.shape)

        # juntar predicciones [rows, models, probabilities]
        prediccion_oneHot = np.expand_dims(prediccion_oneHot, axis=1)
        print('prediccion after', prediccion_oneHot.shape)

        if stack_pred is None:
            stack_pred = prediccion_oneHot

        else:
            print('stack_pred', stack_pred.shape)
            stack_pred = np.hstack((stack_pred, prediccion_oneHot))

    return stack_pred

def return_stacked_dataset_classics(members, inputX):
    stack_pred = None
    for i,model in enumerate(members):
        print(i)
        print("inputX", inputX.shape)

        """
        inputX_2d = inputX.reshape((inputX.shape[0],inputX.shape[1]*inputX.shape[2]))
        print("inputX_2d", inputX_2d.shape)
        """
        # hacer predicciones
        prediccion = model.predict(inputX)

        print('prediccion', prediccion.shape)

        prediccion_oneHot = []
        for i in range(len(prediccion)):
            prediccion_oneHot.append(tf.keras.utils.to_categorical(prediccion[i], num_classes=4))
        prediccion_oneHot = np.array(prediccion_oneHot)

        print('prediccion oneHot', prediccion_oneHot.shape)

        # juntar predicciones [rows, members, probabilities]
        prediccion_oneHot = np.expand_dims(prediccion_oneHot, axis=1)
        print('prediccion after', prediccion_oneHot.shape)

        if stack_pred is None:
            stack_pred = prediccion_oneHot

        else:
            print('stack_pred', stack_pred.shape)
            stack_pred = np.hstack((stack_pred, prediccion_oneHot))

    print("finishd pred", stack_pred.shape)
    return stack_pred


# fit a model based on the outputs from the ensemble members
def fit_stacked_model_Logistic(members, inputX, inputY):
    history = []

    # create dataset using ensemble
    dataset_pred = return_stacked_dataset(members, inputX)
    # reformar vector a 2d para ocupar en logisticRegression
    dataset_pred = dataset_pred.reshape((dataset_pred.shape[0], dataset_pred.shape[1]*dataset_pred.shape[2]))
    
    # fit standalone model
    model = LogisticRegression(multi_class='ovr')

    history.append(model.fit(dataset_pred, inputY))
    return model, history


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(
    models
    , modelsClassic
    , inputX
    , inputXClassic
    , inputY
    , _epochs
    , num_categories=4):


    history = []
    X_test_sub_2d = inputX.reshape((inputX.shape[0],inputX.shape[1]*inputX.shape[2]))

    # crear dataset para ensamble
    dataset_pred = return_stacked_dataset(models, inputX, X_test_sub_2d)

    #convierte el las labels a one-hot vector
    inputY_oneHot = []
    for i in range(len(inputY)):
        inputY_oneHot.append(tf.keras.utils.to_categorical(inputY[i], num_classes=4))
    inputY_oneHot = np.array(inputY_oneHot)

    # fit standalone model
    """ esto es para las no maquinas clasicas"""
    model = create_model_LSTM(
        shape = (dataset_pred.shape[1],dataset_pred.shape[2])
        , _num_categories = num_categories
        , _units = 256
        , _dropout = 0.3
    )
    

    history = model.fit(dataset_pred, inputY_oneHot, batch_size=128, epochs=_epochs)
    return model, history

def fit_stacked_model_classic(
    members
    , inputX
    , inputY
    , _epochs
    , num_categories=4):


    history = []
    print("members", members)
    print("inputX.shape", inputX.shape)
    print("inputY.shape", inputY.shape)


    # crear dataset para ensamble
    #dataset_pred = return_stacked_dataset(members, inputX)

    dataset_pred = return_stacked_dataset_classics(members, inputX)

    print(dataset_pred)
    #convierte el las labels a one-hot vector
    inputY_oneHot = []
    for i in range(len(inputY)):
        inputY_oneHot.append(tf.keras.utils.to_categorical(inputY[i], num_classes=4))
    inputY_oneHot = np.array(inputY_oneHot)

    dataset_pred_oneHot = []
    for i in range(len(dataset_pred)):
        dataset_pred_oneHot.append(tf.keras.utils.to_categorical(dataset_pred[i], num_classes=4))
    dataset_pred_oneHot = np.array(dataset_pred_oneHot)

    print("dataset shape", dataset_pred)
    print("dataset shape", dataset_pred.shape)

    # fit standalone model
    """ esto es para las no maquinas clasicas"""
    model = create_model_LSTM(
        shape = (dataset_pred.shape[1],dataset_pred.shape[2])
        , _num_categories = num_categories
        , _units = 256
        , _dropout = 0.3
    )

    """
    model = create_model_LSTM(
        shape = (dataset_pred_oneHot.shape[0])
        , _num_categories = num_categories
        , _units = 256
        , _dropout = 0.3
    )
    """


    history = model.fit(dataset_pred, inputY_oneHot, batch_size=128, epochs=_epochs)
    return model, history

def stacked_prediction(members, model, inputX):

    X_test_sub_2d = inputX.reshape((inputX.shape[0],inputX.shape[1]*inputX.shape[2]))

    # create dataset using ensemble
    dataset = return_stacked_dataset(members, inputX, X_test_sub_2d)
    
    # make a prediction
    yhat = model.predict(dataset)
    return yhat

def stacked_prediction_classic(members, model, inputX):
    # create dataset using ensemble
    dataset = return_stacked_dataset_classics(members, inputX)
    print(dataset.shape)
    # make a prediction
    yhat = model.predict(dataset)
    return yhat

def creacionBoostingEnsamble(dataset_train, truths_train):
    x = np.zeros((dataset_train.shape[0],dataset_train.shape[1]*dataset_train.shape[2]))

    print("dataset shape", dataset_train.shape)
    print("x shape", x.shape)

    #transforma la forma del vector x e y para ser ocupado en Adaboost 
    x = np.reshape(dataset_train,(dataset_train.shape[0],dataset_train.shape[1]*dataset_train.shape[2]))

    y = np.zeros(len(truths_train))
    for i in range(len(truths_train)):
        y[i]=(np.argmax(truths_train[i]))
        

    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=50,
                            learning_rate=1)
    # Train Adaboost Classifer
    model = abc.fit(x, y)

    return model