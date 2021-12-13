import gensim
import numpy as np
import nltk
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from nltk.corpus import stopwords
from modelos import *
from preprocesamiento import *

#nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle


# ### Problema: Determinar la veracidad de tweets, utilizando clasificación
# - Implementar un clasificador de post de tweets para validar su veracidad.
# - Para esto se utilizará redes neuronales y el árbol de  propagación de los tweets.
# 

# # Parte 1: Procesamiento de datos y funcionalidades
# ## Parte 1.1: Cargar tweets para generar datos de entrenamiento


def normalizarTexto(docText):
    # En gensim.utils, pasa a minúsculas, descarta palabras muy grandes o muy pequeñas.
    return simple_preprocess(docText)



words_not_in_model = list()
def computeDocumentAWE(docText, _model, _model_vocab, _emb_size):
    """
    Calcula el AWE del texto recibido en el parámetro docText.
    La variable docSum almacena la suma de los embeddings de cada
    palabra w en docText. Luego esta suma se divide por el total
    de n palabras consideradas.
    
    Se considero una palabra para el cálculo sólo si esta pertenece
    al vocabulario del modelo. Si no, no es considerada en la suma 
    ni tampoco en el calculo de n.
    """    
    docSum = np.zeros(_emb_size)
    n = 0
    
    ####
    ## AWE = 1/n * Sum w_embedding, para cada w en docText
    ####
    
    normalizedDocText = normalizarTexto(docText)
    
    for w in normalizedDocText:
        ## Se descartan palabras que no están en el modelo de embeddings (vocabulario)
        if w in _model_vocab:
            n = n + 1
            w_embedding = _model.wv[w]
            docSum = docSum + w_embedding
        else:
            words_not_in_model.append(w)

    return docSum / n if n > 0 else docSum    



## Cada palabra tiene un embedding que viene del modelo
## Se calcula AWE para cada post del árbol de propagación (lista de propagación que el primer elemento es el tweet original)
def computeTreeAWE(tree, _model, _model_vocab, _emb_size, all_posts):
    return list(map(lambda t: [t[0], computeDocumentAWE(all_posts[t[1]]['text'],
                                                        _model, _model_vocab, _emb_size), t[2]], tree))



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

## padding al final, con empty
def padAWE(empty, max_num, seq):
    from itertools import repeat
    seq.extend(repeat(empty, max_num - len(seq)))
    return seq

## 
def generate_w2v_variant(_model, _model_vocab, _emb_size, number_of_tweets, tree_max_num_seq, labeled_posts, all_posts):
    empty_awe = np.zeros(_emb_size)
    ## Calcula AWE de cada árbol
    labeled_posts_awe = { k: (v[0], computeTreeAWE(v[1], _model, _model_vocab, _emb_size, all_posts)) for k, v in labeled_posts.items() }
    ## Realiza padding a las secuencias
    padded_labeled_posts_awe = {k: (v[0], padAWE(empty_awe, tree_max_num_seq, v[1])) for k, v in labeled_posts_awe.items()}

    #Genera los datos X e Y para alimentar el modelo de red neuronal
    #Inicialmente con ceros y con la forma adecuada.
    X = np.zeros(shape=(number_of_tweets, tree_max_num_seq, _emb_size)).astype(np.float32)
    Y = np.zeros(shape=(number_of_tweets, num_categories)).astype(np.float32)

    # Asigna al vector X los datos correspondientes
    for idx, (tweet_id, tweet_data) in enumerate(list(padded_labeled_posts_awe.items())):
        for jdx, tweet_d in enumerate(tweet_data[1]):
            ### tweet_d = [uid, tweet_awe, time]
            if jdx == tree_max_num_seq:
                break            
            else:            
                X[idx, jdx, :] = tweet_d[1]

    # Asigna al vector Y los datos correspondientes            
    for idx, (tweet_id, tweet_data) in enumerate(list(padded_labeled_posts_awe.items())):
        Y[idx, :] = to_category_vector(tweet_data[0])

    print(np.shape(X))
    print(np.shape(Y))
    return X, Y



def crearModeloWord2Vec(data):

    # Esto carga la variable con los tweets ya parseados,
    # si se quiere parsear otro dataset ocupar script CargarTweets.py

    f = open('parsedTweets.pckl', 'rb')
    seqs_lens, labeled_posts, all_posts, number_of_tweets = pickle.load(f)
    f.close()

    prefijo = "noPre"

    ###Aqui aplicar los preprocesamientos desde "preprocesamiento.py"

    if(data == 1):
        prefijo = "pre1"

        #Grupo 1
        #P1
        for post in all_posts:
            clearText = replaceUrl(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})

        #P2
        for post in all_posts:
            clearText = replaceRef(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})


    if(data == 2):
        prefijo = "pre2"
        #Grupo 2
        #P3
        for post in all_posts:
            clearText = removeHashtagSymbol(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})

        #P4
        for post in all_posts:
            clearText = removeSpecialChars(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})

        #P5
        for post in all_posts:
            clearText = allLowerCase(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})

        #P6
        for post in all_posts:
            clearText = removeStopWords(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})

        #P7
        for post in all_posts:
            clearText = steeming(all_posts[post]['text'])
            all_posts[post].update({'text':clearText})



    #La red neuronal necesita un tamaño fijo para la secuencia (datos de entrada)
    #¿Que largo de secuencia utilizar?
    counts = np.bincount(seqs_lens) ## seqs_len sólo de los 753
    mean_seq_len = int(np.mean(seqs_lens))
    tree_max_num_seq = mean_seq_len

    # build vocabulary and train model
    w2v50_emb_size = 50
    WINDOW = 5
    W2V_EPOCHS = 50

    documents = []
    for k, v in labeled_posts.items():
        for t in v[1]:
            documents.append(simple_preprocess(all_posts[t[1]]['text']))


    w2v50_model = gensim.models.word2vec.Word2Vec(
    documents,
    vector_size=w2v50_emb_size,
    window=WINDOW,
    min_count=2,
    workers=1,
    epochs=W2V_EPOCHS
    )

    #Train model
    w2v50_model.train(documents, total_examples=len(documents), epochs=w2v50_model.epochs)


    from gensim.models import Word2Vec
    w2v50_model.save(prefijo+"_word2vec.model")
    w2v50_model = Word2Vec.load(prefijo+"_word2vec.model")

    w2v50_model_vocab = w2v50_model.wv.key_to_index



    Xw2v50_full, Yw2v50_full = generate_w2v_variant(
        w2v50_model, 
        w2v50_model_vocab, 
        w2v50_emb_size, 
        number_of_tweets, 
        all_posts=all_posts, 
        tree_max_num_seq = tree_max_num_seq, 
        labeled_posts = labeled_posts
    )

    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(Xw2v50_full, Yw2v50_full, test_size=0.15)
    X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X_train_1, Y_train_1, test_size=0.15)

    f = open('preProcDataset_full.pckl', 'wb')
    pickle.dump([Xw2v50_full, Yw2v50_full, tree_max_num_seq, w2v50_emb_size], f)
    f.close()

    f = open('preProcDataset_sub.pckl', 'wb')
    pickle.dump([X_train_1, X_test_1, Y_train_1, Y_test_1, tree_max_num_seq, w2v50_emb_size], f)
    f.close()

    f = open('preProcDataset_sub_sub.pckl', 'wb')
    pickle.dump([X_train_2, X_test_2, Y_train_2, Y_test_2, tree_max_num_seq, w2v50_emb_size], f)
    f.close()