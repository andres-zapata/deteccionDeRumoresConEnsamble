import nltk
import re
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
nltk.download('stopwords')

#Grupo 1

#P1 Reemplazar las URL con “URL”
def replaceUrl(text):
    text = text.split();
    for i in range(len(text)):
        if (text[i].find('http') != -1):
            text[i] = 'URL'
    return ' '.join(text)


#P2 Remplazar menciones con string 'REF'
def replaceRef(text):
    text = text.split();
    for i in range(len(text)):
        if (text[i].find('@') != -1):
            text[i] = 'REF'
    return(' '.join(text))

#Grupo 2

#P3 quitar símbolo "#" a los hashtags
def removeHashtagSymbol(text):
    return re.sub("[#]","",text) 


#P4 Quita caracteres especiales y emojis
def removeSpecialChars(text):
    regex_pattern = re.compile(pattern = "["
            u"\U00000021-\U00000022"  # ! and ""
            u"\U00000024-\U0000002F"  # punctuation
            u"\U0000003A-\U0000003F"  # punctuation
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'',text)

#P5 Convertir todos los tweets a minusculas.
def allLowerCase(text):
    return text.lower()


#P6 Quitar stop words
def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return(' '.join(filtered_sentence))


#P7 Steeming and lemmanization
def steeming(text):
    ps = PorterStemmer()  
    word_tokens = word_tokenize(text)
    filtered_sentence = []

    for w in word_tokens:
        filtered_sentence.append(ps.stem(w))
    return(' '.join(filtered_sentence))

from nltk.stem import WordNetLemmatizer

def lemmatization(text):
    filtered_sentence = []
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        filtered_sentence.append(lemmatizer.lemmatize(w))
    return(' '.join(filtered_sentence))

def parseTwitterTree(tree_file):
    tree_data = list()
    for line in tree_file:
        _, second_part = line.split('->')
        second_part = second_part.rstrip()
        second_part = second_part.replace("'", "\"")
        tree_data.append(json.loads(second_part))     
    return tree_data

def cargarTweets(ruta_posts, ruta_labels, ruta_tree):
    ### Obtener diccionario con todos los posts
    all_posts = {}
    for file in os.listdir(ruta_posts):
        if file.endswith(".json"):        
            try:        
                with open(os.path.join(ruta_posts, file), 'r') as f:
                    tweet_id  = os.path.splitext(file)[0]
                    tweet_dic = json.load(f)
                    all_posts[tweet_id] = tweet_dic
            except:
                pass

    ### Obtener ids de tweets etiquetados
    labels = {}
    with open(ruta_labels) as label_f:
        for label_line in label_f:
            label, tweet_id = label_line.split(':')
            tweet_id = tweet_id.rstrip()
            labels[tweet_id] = label

    print("Tweets etiquetados      : ", len(labels))        

    seqs_lens = []
    labeled_posts = {}
    number_of_tweets = 0
    number_of_retweets = 0
    number_of_invalid_tweets = 0
    no_in_data = 0
    for tweet_id in labels.keys():
        try:
            if tweet_id in all_posts:
                tree_path = os.path.join(ruta_tree, tweet_id + '.txt')
                with open(tree_path) as tree_file:
                    tree_data = parseTwitterTree(tree_file)
                    
                    ### Remover retweets                
                    first = tree_data[0]
                    without_rt = list(filter(lambda t: t[1] != tweet_id, tree_data))
                    number_of_retweets = number_of_retweets + (len(tree_data) - len(without_rt))
                    only_valid = list(filter(lambda t: t[1] in all_posts, without_rt))
                    number_of_invalid_tweets = number_of_invalid_tweets + (len(without_rt) - len(only_valid))
                    seqs_lens.append(len(only_valid))
                    labeled_posts[tweet_id] = (labels[tweet_id], [first] + only_valid)
                    number_of_tweets = number_of_tweets + 1                
            else:
                no_in_data = no_in_data + 1  
            
        except Exception as e:
            print(e)

            
    print("no_in_data              : ", no_in_data) ## están etiquetados, pero no en los post
    print("number_of_tweets        : ", number_of_tweets)        
    print("all_posts               : ", len(all_posts))
    print("number_of_retweets      : ", number_of_retweets) ## En árbol de propagación
    print("number_of_invalid_tweets: ", number_of_invalid_tweets) ## En árbol de propagación

    return seqs_lens, labeled_posts, all_posts , number_of_tweets
