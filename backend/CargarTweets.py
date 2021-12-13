import pickle
from preprocesamiento import cargarTweets

def requestCargarTweets():
    seqs_lens, labeled_posts, all_posts, number_of_tweets = cargarTweets(
        ruta_posts = "./post", 
        ruta_labels = "./label.txt", 
        ruta_tree = "./tree"
    )

    f = open('parsedTweets.pckl', 'wb')
    pickle.dump([seqs_lens, labeled_posts, all_posts, number_of_tweets], f)
    f.close()