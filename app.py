from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Read the csv file into dataframe df    
    df = pd.read_csv("train.csv")
    n = 159571 #number of records in file
    s = 25000 #desired sample size
    filename = "train.csv"
    skip = sorted(random.sample(range(n),n-s))
    df = pd.read_csv(filename, skiprows=skip)
    df.columns = ["id", "message","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    df = df.reindex(np.random.permutation(df.index))

    comment = df['message']
    comment = comment.as_matrix()

    label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
    label = label.as_matrix()

    comments = []
    labels = []

    for ix in range(comment.shape[0]):
        if len(comment[ix])<=400:
            comments.append(comment[ix])
            labels.append(label[ix])

    labels = np.asarray(labels)
    import string
    print(string.punctuation)
    punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
    print (punctuation_edit)
    outtab = "                                         "
    trantab = str.maketrans(punctuation_edit, outtab)

    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    stop_words = stopwords.words("english")
    for x in range(ord('b'), ord('z')+1):
        stop_words.append(chr(x))

    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    #create objects for stemmer and lemmatizer
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    #download words from wordnet library
    nltk.download('wordnet')

    for i in range(len(comments)):
        comments[i] = comments[i].lower().translate(trantab)
        l = []
        for word in comments[i].split():
            l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
        comments[i] = " ".join(l)

    #import required library
    from sklearn.feature_extraction.text import CountVectorizer

    #create object supplying our custom stop words
    count_vector = CountVectorizer(stop_words=stop_words)
    #fitting it to converts comments into bag of words format
    tf = count_vector.fit_transform(comments)

    # print(count_vector.get_feature_names())
    print(tf.shape)
    def shuffle(matrix, target, test_proportion):
        ratio = int(matrix.shape[0]/test_proportion)
        X_train = matrix[ratio:,:]
        X_test =  matrix[:ratio,:]
        Y_train = target[ratio:,:]
        Y_test =  target[:ratio,:]
        return X_train, X_test, Y_train, Y_test

    X_train, X_test, Y_train, Y_test = shuffle(tf, labels,3)

    from sklearn.naive_bayes import MultinomialNB

    # clf will be the list of the classifiers for all the 6 labels
    # each classifier is fit with the training data and corresponding classifier
    clf = []
    for ix in range(6):
        clf.append(MultinomialNB())
        clf[ix].fit(X_train,Y_train[:,ix])
        
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = count_vector.transform(data)
        my_prediction = []
        for ix in range(6):
            my_prediction.append(clf[ix].predict(vect)[0])
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
