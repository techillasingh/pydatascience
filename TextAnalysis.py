import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer                        # to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text
from sklearn.feature_extraction.text import TfidfVectorizer                        # to transform text into a term and document frequency based representation of numbers
import nltk                                                                        # platform for building Python programs to process natural language
nltk.download('stopwords')                                                         # to download the stop words
nltk.download('punkt')                                                             # tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences
from nltk.corpus import stopwords                                                  # importing the NTLK stopwords to remove articles, preposition and other words that are not actionable
from nltk.stem.porter import PorterStemmer                                         # process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
from wordcloud import WordCloud                                                    # visualization of words based on their frequency
from nltk.tokenize import word_tokenize                                            # allows to create individual objects from a bag of words
from bs4 import BeautifulSoup                                                      # Python library for pulling data from HTML and XML files
import re                                                                          # regular expression (or RE) specifies a set of strings that matches it
from sklearn.naive_bayes import MultinomialNB                                      # to import multinomial naive bayes which is suitable for classification with discrete features (e.g., word counts for text classification)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score  # to import metrics for evaluating the classification model
from sklearn.model_selection import train_test_split
import gradio
import warnings
warnings.filterwarnings('ignore')


# read the dataset
df = pd.read_csv('IMDB_Dataset.csv')
print(df.shape)
print(df.head(10))     # first 10 rows

# summary of the dataset
print(df.describe())

# sentiment count
print(df['sentiment'].value_counts())

# **** Text Cleaning ****

# removing the html strips
def strip_html(text):
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# removing the square brackets
print('remove_between_square_brackets')
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# removing the noisy text
print('denoise_text')
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
# apply function on review column
df['review'] = df['review'].apply(denoise_text)

#removing special characters
# define function for removing special characters
print('remove_special_characters')
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    return text
# apply function on review column
df['review'] = df['review'].apply(remove_special_characters)

#Text Stemming
# stemming the text
print('simple_stemmer')
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
# apply function on review column
df['review'] = df['review'].apply(simple_stemmer)

#Removing stopwords
# setting english stopwords
print('stopword list')
stopword_list = nltk.corpus.stopwords.words('english')
print(stopword_list)

# set stopwords to english
stop = set(stopwords.words('english'))
print(stop)

# removing the stopwords
print('remove_stopwords')
def remove_stopwords(text, is_lower_case=False):
    # splitting strings into tokens (list of words)
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
# apply function on review column
df['review'] = df['review'].apply(remove_stopwords)

print('call train_test_split')
X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Word cloud for positive review words
plt.figure(figsize=(10,10))
df_positive_review =  df[df['sentiment']=='positive']
positive_text = ' '.join(review for review in df_positive_review.review)
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
positive_words = WC.generate(positive_text)
plt.imshow(positive_words, interpolation='bilinear')
plt.show()

# Word cloud for negative review words
plt.figure(figsize=(10,10))
df_negative_review =  df[df['sentiment']=='negative']
negative_text = ' '.join(review for review in df_negative_review.review)
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
negative_words = WC.generate(negative_text)
plt.imshow(negative_words, interpolation='bilinear')
plt.show()

# Count vectorizer
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
# transformed train reviews
cv_train_reviews = cv.fit_transform(X_train)
# transformed test reviews
cv_test_reviews = cv.transform(X_test)
print('CV_train:', cv_train_reviews.shape)
print('CV_test:', cv_test_reviews.shape)

# tfidf vectorizer
tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range = (1,3))
#transformed train reviews
tfidf_train_reviews = tv.fit_transform(X_train)
#transformed test reviews
tfidf_test_reviews = tv.transform(X_test)
print('Tfidf_train:', tfidf_train_reviews.shape)
print('Tfidf_test:', tfidf_test_reviews.shape)

# training the model
mnb=MultinomialNB()
# fitting the NaiveBayes for count vectorizer
mnb_cv = mnb.fit(cv_train_reviews, y_train)
print('MultinomialNB for Count Vectorizer :',mnb_cv)
# fitting the NaiveBayes for tfidf features
mnb_tfidf = mnb.fit(tfidf_train_reviews, y_train)
print('MultinomialNB for tf-idf :',mnb_tfidf)

# predicting the model for CountVectorizer
mnb_cv_predict = mnb.predict(cv_test_reviews)
print('predictions for Count Vectorizer :', mnb_cv_predict)

# predicting the model for tfidf features
mnb_tfidf_predict = mnb.predict(tfidf_test_reviews)
print('predictions for tf-idf :', mnb_tfidf_predict)

# accuracy score for count vectorizer
mnb_cv_score = accuracy_score(y_test, mnb_cv_predict)
print("mnb_cv_score :", mnb_cv_score)

# accuracy score for tf-idf
mnb_tfidf_score = accuracy_score(y_test, mnb_tfidf_predict)
print("mnb_tfidf_score :", mnb_tfidf_score)

# confusion matrix for count vectorizer
cm_cv = confusion_matrix(y_test, mnb_cv_predict, labels=['positive', 'negative'])
print('confusion matrix for count vectorizer :\n', cm_cv)

# confusion matrix for tf-idf
cm_tfidf = confusion_matrix(y_test,mnb_tfidf_predict, labels=['positive','negative'])
print('confusion matrix for tf-idf :\n', cm_tfidf)


# Function for preprocessing of text

def preprocess_text(text):

    text = denoise_text(text)
    text = remove_special_characters(text)
    text = simple_stemmer(text)
    text = remove_stopwords(text)

    return text


# Function to predict label for a review

def predict_review_label(text, vectorizer_method):
    processed_text = preprocess_text(text)

    if vectorizer_method == 'CountVectorizer':
        review = cv.transform([processed_text])
        pred = mnb_cv.predict(review)
    if vectorizer_method == 'TFIDFVectorizer':
        review = tv.transform([processed_text])
        pred = mnb_tfidf.predict(review)
    else:
        review = cv.transform([processed_text])
        pred = mnb_cv.predict(review)

    return pred[0]


# Testing a review
print(predict_review_label("It was a good movie, really enjoyed it a lot.", 'CountVectorizer'))
# Testing a review
print(predict_review_label("Was very bad, I was barely able to understand the concepts.", 'TFIDFVectorizer'))


