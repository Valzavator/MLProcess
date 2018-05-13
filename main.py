from ML_process import prepare_messages, vectorize_messages, clusterize, build_clouds
from database import Database


def main():
    count_of_clusters = 9
    db = Database()
    prepared_messages = prepare_messages(db.get_text_from_messages())
    matrix = vectorize_messages(prepared_messages)
    model = clusterize(count_of_clusters, matrix)
    build_clouds(model, count_of_clusters, prepared_messages)
    db.close()


if __name__ == "__main__":
    main()

