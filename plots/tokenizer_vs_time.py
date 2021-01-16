import matplotlib.pyplot as plt



def plot_tokenizers_vs_time(vectorizers_time, classifier):
    plt.figure()
    plt.bar(LABELS, vectorizers_time, width=0.3)
    plt.title("Train and test time per Vectorizer for '{}' classifier".format(classifier))
    plt.ylabel("Train and test time (secs)")
    plt.yscale('log')
    plt.show()



if __name__ == "__main__":
    LABELS = ['TfIdfVectorizer', 'TfIdfVectorizer (Lemmatize)', 'TfIdfVectorizer (Stemming)']

    # Naive Bayes classifier
    naive_bayes_vectorizers_time = [0.9424, 30.7023, 235.69]
    plot_tokenizers_vs_time(naive_bayes_vectorizers_time, "Naive Bayes")

    # Logistic regression classifier
    logistic_vectorizers_time = [1.7131, 35.9588, 226.3316]
    plot_tokenizers_vs_time(logistic_vectorizers_time, "Logistic Regression")

    # Linear SVM
    svm_vectorizers_time = [2.4967, 32.4222, 225.6811]
    plot_tokenizers_vs_time(svm_vectorizers_time, "Linear SVM")
