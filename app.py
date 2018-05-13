import nltk
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

nltk.download('stopwords')


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
    word_cloud = WordCloud(background_color='white',
                           width=1200,
                           height=1000
                           ).generate(text)
    # plt.imshow(word_cloud)
    # plt.axis('off')
    # plt.show()

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(word_cloud)
    f.add_subplot(1, 2, 2)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show(block=True)

