import nltk
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from database import Database
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

db = Database()
messages1 = ["This is  will very, ,,,will ,,,123 ...strange",
             "This is  will very is is is is is is nice cs",
             "My name is ,is is is, is is is is is is is cs nice",
             "My name will is cs will  Vadim and ,I`d like to play dota go",
             "This is will very pretty doing do reading girl cs"]


# messages1 = db.get_text_from_messages()

def prepare_messages(messages: list) -> list:
    lemmatizer = nltk.WordNetLemmatizer()
    processed = []
    for message in messages:
        words = word_tokenize(str.lower(message))
        tokens = [lemmatizer.lemmatize(word) for word in words
                  if word.isalpha() and word not in stopwords.words("english")]
        if len(tokens) > 0:
            processed.append(tokens)
    return [' '.join(word for word in sentences) for sentences in processed]


def vectorize_messages(messages: list):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(messages)


def clusterize(number_of_clusters: int, matrix: scipy.matrix):
    model = KMeans(n_clusters=number_of_clusters, random_state=0, n_jobs=-2)
    model.fit(matrix)
    return model.predict(matrix)


def one_cloud(model, messages):
    text = ""
    for idx, cluster in enumerate(model):
        print(cluster, messages[idx])
        text += messages[idx] + " "

    print(text)

    word_cloud = WordCloud(background_color='white',
                           width=1200,
                           height=1000
                           ).generate(text)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()


num_clusters = 2
prepared_messages = prepare_messages(messages1)
matrix = vectorize_messages(prepared_messages)
model = clusterize(num_clusters, matrix)
one_cloud(model, prepared_messages)


