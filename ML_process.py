import math
import matplotlib.pyplot as plt
import nltk
import scipy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
from wordcloud import WordCloud

nltk.download('stopwords')


def prepare_messages(messages: list) -> list:
    processed = []
    for message in messages:
        words = word_tokenize(str.lower(message))
        tokens = [Word(word).lemmatize("v") for word in words
                  if word.isalpha() and word not in stopwords.words("english")]
        if len(tokens) > 0:
            print(tokens)
            processed.append(tokens)
    return [' '.join(word for word in sentences) for sentences in processed]


def vectorize_messages(messages: list):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(messages)


def clusterize(count_of_clusters: int, matrix: scipy.matrix):
    model = KMeans(n_clusters=count_of_clusters, random_state=0, n_jobs=-2)
    model.fit(matrix)
    return model.predict(matrix)


def build_clouds(model, count_of_clusters: int, messages):
    result_cloud = plt.figure(figsize=(40, 30))
    for num_of_cluster in range(count_of_clusters):
        text = ""
        print("\nCLUSTER: " + str(num_of_cluster) + "\n")
        for idx, cluster in enumerate(model):
            if num_of_cluster == cluster:
                print(cluster, "-", idx, "-", messages[idx])
                text += messages[idx] + " "
        if len(text) > 0:
            word_cloud = WordCloud(
                background_color='white'
            ).generate(text)
            result_cloud.add_subplot(
                math.ceil(count_of_clusters / 3),
                3,
                num_of_cluster + 1
            )
            plt.axis('off')
            plt.imshow(word_cloud)
    plt.savefig('images/tags_cloud.png')
    plt.show()

