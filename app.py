import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from database import Database
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

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



#
# print(X)
# model = KMeans(n_clusters=k, random_state=0, n_jobs=-2)
# model.fit(X)
# y = model.predict(X)
#
# text = ""
# for idx, cluster in enumerate(y):
#     print(cluster, processed_messages[idx])
#     text += processed_messages[idx] + " "
#
# print(text)
#
# word_cloud = WordCloud(background_color='white',
#                        width=1200,
#                        height=1000
#                        ).generate(text)
# plt.imshow(word_cloud)
# plt.axis('off')
# plt.show()
