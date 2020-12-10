from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from models import TextClassification
from extract import Extractor


data_fields = ["reviewText"]
target_field = "overall"

df = Extractor.extract_examples("../data/Software_5.json", data_fields=data_fields, target_field=target_field)
df.drop_duplicates(inplace=True)

X = df.reviewText
y = df.overall

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

vectorizer = TfidfVectorizer()
classifier = LinearSVC()

classification_task = TextClassification(vectorizer, classifier)
classification_task.fit(X_train, y_train)
y_predicted = classification_task.predict(X_test)

print(classification_task.classification_report(y_test, y_predicted))
