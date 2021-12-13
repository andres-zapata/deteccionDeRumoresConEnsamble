import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Bidirectional, Conv1D, Flatten, MaxPooling1D

### tensorflow OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
###################################################################################################
def create_model_LSTM(    
        shape            
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(LSTM(_units, input_shape = shape, return_sequences=False))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    #model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

###################################################################################################
def create_model_StackedLSTM(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(LSTM(_units, input_shape=shape, return_sequences=True))
    model.add(LSTM(_units, return_sequences=False))        
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

###################################################################################################
def create_model_GRU(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(GRU(_units, input_shape=shape, return_sequences=False))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###################################################################################################
def create_model_StackedGRU(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(GRU(_units, input_shape=shape, return_sequences=True))
    model.add(GRU(_units, return_sequences=False))        
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###################################################################################################
def create_model_BI_LSTM(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(Bidirectional(LSTM(_units, input_shape=shape, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

###################################################################################################
def create_model_BI_StackedLSTM(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(Bidirectional(LSTM(_units, input_shape=shape, return_sequences=True)))
    model.add(Bidirectional(LSTM(_units, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model
    
###################################################################################################
def create_model_BI_GRU(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(Bidirectional(GRU(_units, input_shape=shape, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###################################################################################################
def create_model_BI_StackedGRU(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3):
    
    model = Sequential()
    model.add(Bidirectional(GRU(_units, input_shape=shape, return_sequences=True)))
    model.add(Bidirectional(GRU(_units, return_sequences=False)))
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###################################################################################################
def create_model_Conv1D(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3
        , _kernel_size = 2): # Cambiar a 1
    
    model = Sequential()
    model.add(Conv1D(_units, _kernel_size, activation='relu', input_shape=shape))
    model.add(MaxPooling1D())
    model.add(Conv1D(_units, _kernel_size, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(_dropout))
    model.add(Dense(_num_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###################################################################################################
def create_model_RCNN(
        shape
        , _num_categories
        , _units = 200
        , _dropout = 0.3
        , _kernel_size = 2):    

    model = Sequential()
    model.add(Conv1D(_units, _kernel_size, activation='relu', input_shape=shape))
    model.add(MaxPooling1D())
    model.add(Conv1D(_units, _kernel_size, activation='relu'))
    model.add(MaxPooling1D())
    model.add(LSTM(_units, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(_units, recurrent_dropout=0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(_num_categories, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model
