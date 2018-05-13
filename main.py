from app import prepare_messages, vectorize_messages, clusterize, one_cloud
from database import Database

if __name__ == "__main__":
    db = Database()
    num_clusters = 2
    prepared_messages = prepare_messages(db.get_text_from_messages())
    matrix1 = vectorize_messages(prepared_messages)
    model1 = clusterize(num_clusters, matrix1)
    one_cloud(model1, prepared_messages)
