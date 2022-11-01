from flask import Flask, request, render_template, Response
import pandas as pd
import re
application = Flask(__name__)
#################################################################################
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import pickle
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

model_cv= pickle.load(open('model2_cv.pkl','rb'))
cv = pickle.load(open('cv.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
model_tfidf= pickle.load(open('model2_tfidf.pkl','rb'))
confusion_tfidf=pickle.load(open('tfidf_accuracy.pkl','rb'))
confusion_cv=pickle.load(open('cv_accuracy.pkl','rb'))

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001

    token_features = [0.0] * 4

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[1] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[2] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[3] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 2

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def query_point_creator_cv(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()

    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 17), q1_bow, q2_bow))


def query_point_creator_tfidf(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # bow feature for q1
    q1_bow = tfidf.transform([q1]).toarray()

    # bow feature for q2
    q2_bow = tfidf.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 17), q1_bow, q2_bow))

def predict_function(question1, question2, feature_extracion, ml_algorithm):
    res = -1
    if(feature_extracion=='bow'):
        if(ml_algorithm=="random_forest"):
            res = model_cv[0].predict(query_point_creator_cv(question1, question2))[0]
        elif(ml_algorithm=="decision_tree"):
            res = model_cv[1].predict(query_point_creator_cv(question1, question2))[0]
        elif(ml_algorithm=="logistic_regression"):
            res = model_cv[2].predict(query_point_creator_cv(question1, question2))[0]
        else:
            res = model_cv[3].predict(query_point_creator_cv(question1, question2))[0]
    else:
        if(ml_algorithm=="random_forest"):
            res = model_tfidf[0].predict(query_point_creator_tfidf(question1, question2))[0]
        elif(ml_algorithm=="decision_tree"):
            res = model_tfidf[1].predict(query_point_creator_tfidf(question1, question2))[0]
        elif(ml_algorithm=="logistic_regression"):
            res = model_tfidf[2].predict(query_point_creator_tfidf(question1, question2))[0]
        else:
            res = model_tfidf[3].predict(query_point_creator_tfidf(question1, question2))[0]
    return res

def model_confusion_matrix(metrix, ml_algo):
    if(ml_algo=="random_forest"):
        res = metrix[0]
    elif(ml_algo=="decision_tree"):
        res = metrix[1]
    elif(ml_algo=="logistic_regression"):
        res = metrix[2]
    else:
        res = metrix[3]
    accuracy = (res[0][0]+res[1][1])/(res[0][0]+res[0][1]+res[1][0]+res[1][1])
    precision = res[0][0]/(res[0][0]+res[1][0])
    recall = res[0][0]/(res[0][0]+res[0][1])
    f1score = (2*precision*recall)/(precision+recall)
    result = [accuracy, precision, recall, f1score]
    return result



#########################################################################################################

@application.route('/')
def index():
    return render_template('FrontPage.html')

@application.route('/SignUp', methods =["GET", "POST"])
def signUp():
    if request.method == "POST":
        df = pd.read_csv('database_file.csv')
        name = request.form.get("nm")
        phone = request.form.get("mob")
        email = request.form.get("email")
        password = request.form.get("pass")
        mobpat='[6-9][0-9]{9}'
        emailpat = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"
        passpat= "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{8,18}$"
        '''''
        Please enter password should be
           1) One Capital Letter
           2) Special Character
           3) One Number
           4) Length Should be 8-18:
             ex: asd@Asda
        '''''
        if ((not re.match(emailpat, email)) or (not re.match(mobpat, phone)) or (not re.match(passpat, password))):
            return render_template('popup.html')
        search = df.loc[df['Password'] == password]
        if (name=='' or phone=='' or email=='' or password==''):
            return "Try again"
        elif (len(search)>0):
            return render_template('login.html')
        else:
            data= {'Name': name,'Mobile': phone, 'Email': email, 'Password': password, 'Feedback': ''}
            df = df.append(data, ignore_index=True)
            df.to_csv('database_file.csv', index=False)
    return render_template('SignUp.html')

@application.route('/login', methods =["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("pass")
        pat = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"
        if (not re.match(pat, email)):
            return render_template('popup.html')
        df = pd.read_csv('database_file.csv')
        search = df.loc[df['Password'] == password]
        email1 = df.loc[df['Email']==email]
        if(len(search)==0 or len(email1) == 0 ):
            return render_template('login.html')
        else:
            if(email=="ajaytiwarinitjsr@gmail.com" and password=="Secret@123"):
                return Response(open('database_file.csv', 'rb'), mimetype="text/csv", headers={"Content-disposition":"attachment; filename=database_file.csv"})
            else:
                return render_template('prediction.html', result="No Result", acc=0.0, prec=0.0, rec=0.0,f1score=0.0)
    return render_template('login.html')

@application.route('/forget', methods =["GET", "POST"])
def forget():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("pass")
        passpat = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{8,18}$"
        df = pd.read_csv('database_file.csv')
        search = df.loc[df['Email'] == email]
        if(len(search)==0 or (not re.match(passpat, password))):
            return render_template('popup.html')
        else:
            df.at[df.loc[df['Email'] == email].index[0], 'Password'] = password
            df.to_csv('database_file.csv', index=False)
            return render_template('login.html')
    return render_template('forget.html')

@application.route('/feedback', methods =["GET", "POST"])
def feedback():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("pass")
        textarea1= request.form["text_area"]
        pat = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"
        if (not re.match(pat, email)):
            return render_template('popup.html')
        df = pd.read_csv('database_file.csv')
        search = df.loc[df['Password'] == password]
        email1 = df.loc[df['Email']==email]
        if(len(search)==0 or len(email1) == 0 ):
            return render_template('popup.html')
        else:
            df.at[df.loc[df['Password'] == password].index[0], 'Feedback'] = textarea1
            df.to_csv('database_file.csv', index=False)
            return render_template('FrontPage.html')
    return render_template('Feedback.html')

@application.route('/prediction', methods =["GET", "POST"])
def prediction():
    ret_res = [0.0, 0.0, 0.0, 0.0]
    result = "No Result"
    if request.method == "POST":
        question1 = request.form.get("question1")
        question2 = request.form.get("question2")
        nlp_tech = request.form.get('nlp_fet')
        ml_algorithm = request.form.get('ml_algo')
        res = predict_function(question1, question2, nlp_tech, ml_algorithm)
        if(res==1):
            result = "Duplicate Questions"
        else:
            result = "Not Duplicate Questions"
        ret_res=[]
        if(nlp_tech=='bow'):
            ret_res = model_confusion_matrix(confusion_cv, ml_algorithm)
        else:
            ret_res = model_confusion_matrix(confusion_tfidf, ml_algorithm)
        return render_template('prediction.html', result=result, acc=round(ret_res[0]*100, 2), prec=round(ret_res[1]*100, 2), rec=round(ret_res[2]*100, 2), f1score=round(ret_res[3]*100, 2))
    return render_template('prediction.html', result=result, acc=ret_res[0], prec=ret_res[1], rec=ret_res[2], f1score=ret_res[3])



if __name__ == '__main__':
    application.run()

