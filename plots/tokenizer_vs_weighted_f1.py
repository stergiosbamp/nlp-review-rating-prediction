import matplotlib.pyplot as plt



def plot_tokenizers_vs_time(f1_scores, classifier):
    plt.figure()
    plt.bar(LABELS, f1_scores, width=0.3)
    plt.title("Weighted F1-score per Vectorizer for '{}' classifier".format(classifier))
    plt.ylabel("F1-score")
    # plt.yscale('log')
    plt.show()



if __name__ == "__main__":
    LABELS = ['TfIdfVectorizer', 'TfIdfVectorizer (Lemmatize)', 'TfIdfVectorizer (Stemming)']

    # Naive Bayes classifier
    naive_bayes_f1_scores = [0.4125, 0.3899, 0.3829]
    plot_tokenizers_vs_time(naive_bayes_f1_scores, "Naive Bayes")

    # Logistic regression classifier
    logistic_f1_scores = [0.5337, 0.5303, 0.5314]
    plot_tokenizers_vs_time(logistic_f1_scores, "Logistic Regression")

    # Linear SVM
    svm_f1_scores = [0.5128, 0.5074, 0.5008]
    plot_tokenizers_vs_time(svm_f1_scores, "Linear SVM")
