import argparse

from embeddings import MeanGloveTwitterVectorizer
from models import TextClassification
from experiment import Experiment
from resample import Oversampler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def glove_twitter_100():
    mean_100_vectorizer = MeanGloveTwitterVectorizer(
        model_path='../data/glove-twitter-100.kv',
        model_to_download='glove-twitter-100',
        model_vector_size=100)

    text_clf = TextClassification(mean_100_vectorizer,
                                  LogisticRegression(solver='saga', C=1.5))

    return text_clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment file for SVM")
    parser.add_argument('url', type=str, help="A dataset URL from ")
    args = parser.parse_args()
    url = args.url

    X, y = Experiment.load_data(url, dest_dir="../data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_res, y_train_res = Oversampler().resample(X_train, y_train)

    print("Examining Glove Twitter 100...")
    # Glove Twitter 100
    text_clf_twitter_100 = glove_twitter_100()

    text_clf_twitter_100.fit(X_train_res, y_train_res)

    print(f"Total words: {text_clf_twitter_100.pipeline.steps[0][1].total_words_counter}")
    print(f"Found words: {text_clf_twitter_100.pipeline.steps[0][1].found_words_counter}")
    print(f"Not found words: {text_clf_twitter_100.pipeline.steps[0][1].not_found_words_counter}")
    print(f"Unique total words: {len(text_clf_twitter_100.pipeline.steps[0][1].unique_total_words_counter)}")
    print(f"Unique found words: {len(text_clf_twitter_100.pipeline.steps[0][1].unique_found_words_counter)}")
    print(f"Unique not found words: {len(text_clf_twitter_100.pipeline.steps[0][1].unique_not_found_words_counter)}")
