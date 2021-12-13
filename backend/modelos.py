from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, GRU, Bidirectional, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import StratifiedKFold
import time
import os
import numpy as np

UNITS = 256
DROPOUT = 0.5

def create_model_LSTM( 
        shape
        , num_categories
        , units = UNITS
        , _dropout = DROPOUT
        , _funcionActivacion = 'sigmoid'):

    print("Shape of LSTM - ",shape)

    ##shape = np.reshape((shape))
    model = Sequential()
    model.add(LSTM(units, input_shape=shape, return_sequences=False))
    model.add(Dropout(_dropout))
    model.add(Dense(num_categories))
    model.add(Activation(_funcionActivacion))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

def create_model_StackedLSTM(
        shape
        , _num_categories
        , _units = UNITS
        , _dropout = DROPOUT
        , _funcionActivacion = 'sigmoid'):

    
    model = Sequential()
    model.add(LSTM(_units, input_shape=shape, return_sequences=True))
    model.add(LSTM(_units, return_sequences=False))        
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation(_funcionActivacion))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

def create_model_BI_GRU(
        shape
        , _num_categories
        , _units = UNITS
        , _dropout = DROPOUT
        , _funcionActivacion = 'sigmoid'):
    
    print(shape)
    model = Sequential()
    model.add(Bidirectional(GRU(_units, input_shape=shape, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation(_funcionActivacion))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_RCNN(
        shape
        , _num_categories
        , _units = UNITS
        , _dropout = DROPOUT
        , _kernel_size = 2
        , _funcionActivacion = 'sigmoid'):

    model = Sequential()
    model.add(Conv1D(_units, _kernel_size, activation='relu', input_shape=shape))
    model.add(MaxPooling1D())
    model.add(Conv1D(_units, _kernel_size, activation='relu'))
    model.add(MaxPooling1D())
    model.add(LSTM(_units, return_sequences=True, recurrent_dropout=_dropout))
    model.add(LSTM(_units, recurrent_dropout=_dropout))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(_num_categories, activation=_funcionActivacion))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

def create_model_BI_StackedGRU(
        shape
        , _num_categories
        , _units = UNITS
        , _dropout = DROPOUT
        , _funcionActivacion = 'sigmoid'):
    
    model = Sequential()
    model.add(Bidirectional(GRU(_units, input_shape=shape, return_sequences=True)))
    model.add(Bidirectional(GRU(_units, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation(_funcionActivacion))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def perform_kfold_validation_LSTM(X, Y, _emb_size, tree_max_num_seq, k = 5, _verbose = 0, _epochs=200):
    seed = 7
    results_score = []
    results_acc   = []
    results_time  = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for j, (train_idx, test_idx) in enumerate(kfold.split(X, Y.argmax(1))):
        print('\nFold ',j)
        print("\nlen train index: %s" % len(train_idx))       
        print("len test index: %s" % len(test_idx))
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        X_test_cv  = X[test_idx]
        Y_test_cv  = Y[test_idx]

        model = create_model_LSTM(
            shape=(tree_max_num_seq, _emb_size)
            , num_categories= 4
            , units=UNITS
            , _dropout=DROPOUT
        )
     
        start_time = time.time()
        model.fit(X_train_cv, Y_train_cv, batch_size=128, epochs = _epochs, verbose= _verbose, use_multiprocessing=False)     ## entrenamos
        score, acc = model.evaluate(X_test_cv, Y_test_cv, batch_size=128,verbose= _verbose) ## evaluamos /acc: accuracy
        end_time = time.time()

        print('Score: %1.4f' % score)
        print('Accuracy: %1.4f' % acc)  
        print('time: %1.4f' % (end_time - start_time))
        results_score.append(score)
        results_acc.append(acc)
        results_time.append((end_time - start_time))
    
    return results_score, results_acc, results_time

def perform_kfold_validation_LSTM2(X, Y, _emb_size,  tree_max_num_seq, w2v50_emb_size, k = 5, _verbose = 0, _epochs=200):
    seed = 7
    results_score = []
    results_acc   = []
    results_time  = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for j, (train_idx, test_idx) in enumerate(kfold.split(X, Y.argmax(1))):
        print('\nFold ',j)
        print("\nlen train index: %s" % len(train_idx))       
        print("len test index: %s" % len(test_idx))
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        X_test_cv  = X[test_idx]
        Y_test_cv  = Y[test_idx]

        model = create_model_StackedLSTM(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= 4
            , _units=UNITS
            , _dropout=DROPOUT
        )
     
        start_time = time.time()
        model.fit(X_train_cv, Y_train_cv, batch_size=128, epochs= _epochs, verbose= _verbose, use_multiprocessing=False)     ## entrenamos
        score, acc = model.evaluate(X_test_cv, Y_test_cv, batch_size=128,verbose= _verbose) ## evaluamos /acc: accuracy
        end_time = time.time()

        print('Score: %1.4f' % score)
        print('Accuracy: %1.4f' % acc)  
        print('time: %1.4f' % (end_time - start_time))
        results_score.append(score)
        results_acc.append(acc)
        results_time.append((end_time - start_time))
    
    return results_score, results_acc, results_time

def perform_kfold_validation_RCNN(X, Y, _emb_size, tree_max_num_seq, w2v50_emb_size, k = 5, _verbose = 0, _epochs=200):
    seed = 7
    results_score = []
    results_acc   = []
    results_time  = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for j, (train_idx, test_idx) in enumerate(kfold.split(X, Y.argmax(1))):
        print('\nFold ',j)
        print("\nlen train index: %s" % len(train_idx))       
        print("len test index: %s" % len(test_idx))
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        X_test_cv  = X[test_idx]
        Y_test_cv  = Y[test_idx]

        model = create_model_RCNN(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= 4
            , _units=512
            , _dropout=DROPOUT
        )
     
        start_time = time.time()
        model.fit(X_train_cv, Y_train_cv, batch_size=128, epochs= _epochs, verbose= _verbose, use_multiprocessing=False)     ## entrenamos
        score, acc = model.evaluate(X_test_cv, Y_test_cv, batch_size=128,verbose= _verbose) ## evaluamos /acc: accuracy
        end_time = time.time()

        print('Score: %1.4f' % score)
        print('Accuracy: %1.4f' % acc)  
        print('time: %1.4f' % (end_time - start_time))
        results_score.append(score)
        results_acc.append(acc)
        results_time.append((end_time - start_time))
    
    return results_score, results_acc, results_time

def perform_kfold_validation_BIGRU(X, Y, _emb_size, tree_max_num_seq, w2v50_emb_size, k = 5, _verbose = 0, _epochs=200):
    seed = 7
    results_score = []
    results_acc   = []
    results_time  = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for j, (train_idx, test_idx) in enumerate(kfold.split(X, Y.argmax(1))):
        print('\nFold ',j)
        print("\nlen train index: %s" % len(train_idx))       
        print("len test index: %s" % len(test_idx))
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        X_test_cv  = X[test_idx]
        Y_test_cv  = Y[test_idx]

        model = create_model_BI_GRU(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= 4
            , _units=UNITS
            , _dropout=DROPOUT
        )
     
        start_time = time.time()
        model.fit(X_train_cv, Y_train_cv, batch_size=128, epochs= _epochs, verbose= _verbose, use_multiprocessing=False)     ## entrenamos
        score, acc = model.evaluate(X_test_cv, Y_test_cv, batch_size=128,verbose= _verbose) ## evaluamos /acc: accuracy
        end_time = time.time()

        print('Score: %1.4f' % score)
        print('Accuracy: %1.4f' % acc)  
        print('time: %1.4f' % (end_time - start_time))
        results_score.append(score)
        results_acc.append(acc)
        results_time.append((end_time - start_time))
    
    return results_score, results_acc, results_time

def perform_kfold_validation_bigru2(X, Y, _emb_size, tree_max_num_seq, w2v50_emb_size, k = 5, _verbose = 0, _epochs=200):
    seed = 7
    results_score = []
    results_acc   = []
    results_time  = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for j, (train_idx, test_idx) in enumerate(kfold.split(X, Y.argmax(1))):
        print('\nFold ',j)
        print("\nlen train index: %s" % len(train_idx))       
        print("len test index: %s" % len(test_idx))
        X_train_cv = X[train_idx]
        Y_train_cv = Y[train_idx]
        X_test_cv  = X[test_idx]
        Y_test_cv  = Y[test_idx]

        model = create_model_BI_StackedGRU(
            shape=(tree_max_num_seq, w2v50_emb_size)
            , _num_categories= 4
            , _units=UNITS
            , _dropout=DROPOUT
        )
     
        start_time = time.time()
        model.fit(X_train_cv, Y_train_cv, batch_size=128, epochs= _epochs, verbose= _verbose, use_multiprocessing=False)     ## entrenamos
        score, acc = model.evaluate(X_test_cv, Y_test_cv, batch_size=128,verbose= _verbose) ## evaluamos /acc: accuracy
        end_time = time.time()

        print('Score: %1.4f' % score)
        print('Accuracy: %1.4f' % acc)  
        print('time: %1.4f' % (end_time - start_time))
        results_score.append(score)
        results_acc.append(acc)
        results_time.append((end_time - start_time))
    
    return results_score, results_acc, results_timed


def perform_final_validation_LSTM(
    X
    , Y
    , X_validate
    , Y_validate
    , _emb_size
    , _epochs
    , _dropout
    , _unidadesDimensionales
    , _funcionActivacion
    , tree_max_num_seq 
    ,_verbose=1):

    history = []
    all_models = []
    print("########################")
    print("## MODEL LSTM ")
    
    print("_unidadesDimensionales", _unidadesDimensionales)
    print("_dropout", _dropout)
    modelLSTM = create_model_LSTM(
        shape=(tree_max_num_seq, _emb_size)
        , num_categories=4
        , units = _unidadesDimensionales
        , _dropout= _dropout
        , _funcionActivacion = _funcionActivacion
    )
            
    modelLSTM.summary()
    
    history.append(modelLSTM.fit(X, Y, batch_size=128, epochs=_epochs, verbose=_verbose))
    #score, acc = modelLSTM.evaluate(X_validate, Y_validate, batch_size=128)

    filename='model_LSTM.h5'

    if(os.path.isfile('_'+filename)):
        os.remove('_'+filename)
        os.rename(filename, "_"+filename)

    modelLSTM.save(filename)

    all_models.append(modelLSTM)
    return history, all_models




def perform_final_validation_RCNN(X, Y, X_validate, Y_validate, _emb_size, _epochs, _unidadesDimensionales, _funcionActivacion, _dropout, tree_max_num_seq ,_verbose=1):
    h_units_score = []
    h_units_acc   = []
    history = []
    all_models = []
    
    print("########################")
    print("## MODEL RCNN ")

    modelRCNN = create_model_RCNN(
        shape=(tree_max_num_seq, _emb_size)
        , _num_categories= 4
        , _units = _unidadesDimensionales
        , _dropout= _dropout
        , _funcionActivacion = _funcionActivacion
    )
            
    modelRCNN.build(input_shape=(tree_max_num_seq, _emb_size))
    modelRCNN.summary()

    history.append(modelRCNN.fit(X, Y, batch_size=128, epochs=_epochs, verbose=_verbose))
    score, acc = modelRCNN.evaluate(X_validate, Y_validate, batch_size=128)
    h_units_score.append(score)
    h_units_acc.append(acc)

    filename='model_RCNN.h5'

    if(os.path.isfile('_'+filename)):
        os.remove('_'+filename)
        os.rename(filename, "_"+filename)

    modelRCNN.save(filename)
    all_models.append(modelRCNN)

    return history, all_models



def perform_final_validation_LSTM2(X, Y, X_validate, Y_validate, _emb_size, _epochs, _unidadesDimensionales, _funcionActivacion, _dropout, tree_max_num_seq ,_verbose=1):
    h_units_score = []
    h_units_acc   = []
    history = []
    all_models = []
    
    print("########################")
    print("## MODEL LSTM-2 ")

    modelLSTM2 = create_model_StackedLSTM(
        shape=(tree_max_num_seq, _emb_size)
        , _num_categories=4
        , _units = _unidadesDimensionales
        , _dropout = _dropout
        , _funcionActivacion = _funcionActivacion
    )
            
    modelLSTM2.build(input_shape=(tree_max_num_seq, _emb_size))
    modelLSTM2.summary()
    
    history.append(modelLSTM2.fit(X, Y, batch_size=128, epochs=_epochs, verbose=_verbose))
    score, acc = modelLSTM2.evaluate(X_validate, Y_validate, batch_size=128)
    h_units_score.append(score)
    h_units_acc.append(acc)


    filename='model_LSTM2.h5'

    if(os.path.isfile('_'+filename)):
        os.remove('_'+filename)
        os.rename(filename, "_"+filename)

    modelLSTM2.save(filename)

    all_models.append(modelLSTM2)

    print(history)

    return history, all_models



def perform_final_validation_bigru(X, Y, X_validate, Y_validate, _emb_size, tree_max_num_seq, _epochs, _unidadesDimensionales, _funcionActivacion, _dropout, _verbose=1):
    history = []
    all_models = []
    
    print("########################")
    print("## MODEL bigru ")

    modelBIGRU = create_model_BI_GRU(
        shape=(tree_max_num_seq, _emb_size)
        , _num_categories = 4
        , _units = _unidadesDimensionales
        , _dropout = _dropout
        , _funcionActivacion = _funcionActivacion
    )

    #modelBIGRU.build(input_shape=(tree_max_num_seq, _emb_size))
    #modelBIGRU.summary()
    history.append(modelBIGRU.fit(X, Y, batch_size=128, epochs=_epochs, verbose=_verbose))


    filename='modelBIGRU.h5'
    if(os.path.isfile('_'+filename)):
        os.remove('_'+filename)
        os.rename(filename, "_"+filename)
    modelBIGRU.save(filename)

    all_models.append(modelBIGRU)
    return history, all_models



def perform_final_validation_bigru2(X, Y, X_validate, Y_validate, _emb_size, _epochs, _unidadesDimensionales, _funcionActivacion, _dropout, tree_max_num_seq, _verbose=1):
    history = []
    all_models = []
    
    print("########################")
    print("## MODEL bigru 2")
    modelBIGRU = create_model_BI_StackedGRU(
        shape=(tree_max_num_seq, _emb_size)
        , _num_categories = 4
        , _units = _unidadesDimensionales
        , _dropout = _dropout
        , _funcionActivacion = _funcionActivacion
    )
    
    history.append(modelBIGRU.fit(X, Y, batch_size=128, epochs=_epochs, verbose=_verbose))
    filename='modelBIGRU2.h5'

    if(os.path.isfile('_'+filename)):
        os.remove('_'+filename)
        os.rename(filename, "_"+filename)

    modelBIGRU.save(filename)
    all_models.append(modelBIGRU)

    return history, all_models

def trainNaiveBayes(X, Y):
    from sklearn.naive_bayes import MultinomialNB, GaussianNB

    #reformar los dataset para ocupar en naive bayes
    X_sub_train_2d      = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    Y_sub_train_labeled = np.argmax(Y,1)

    ##Multinomial no puede aceptar valores negativos: trasladarlos o usar GaussianNB
    modelo = GaussianNB() 
    modelo.fit(X_sub_train_2d,Y_sub_train_labeled)

    from joblib import dump
    dump(modelo, 'naiveBayesModel.joblib')


def trainSVM(X, Y):
    from sklearn.svm import SVC

    #reformar los dataset para ocupar en naive bayes
    X_sub_train_2d      = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    Y_sub_train_labeled = np.argmax(Y,1)

    model = SVC(kernel ='linear', C = 1)
    
    model.fit(X_sub_train_2d, Y_sub_train_labeled)

    # >Guardar modelo
    from joblib import dump
    dump(model, 'svmModel.joblib') 


def trainRandomForest(X,Y):
    from sklearn.ensemble import RandomForestClassifier

    #reformar los dataset para ocupar en naive bayes
    X_sub_train_2d      = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    Y_sub_train_labeled = np.argmax(Y,1)

    model = RandomForestClassifier(n_estimators=100, random_state=np.random.RandomState(seed=SEED))

    model.fit(X_sub_train_2d, Y_sub_train_labeled)

    # >Guardar modelo
    from joblib import dump
    dump(model, 'randomForestModel.joblib')