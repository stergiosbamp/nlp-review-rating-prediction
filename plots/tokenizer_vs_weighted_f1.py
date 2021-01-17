import matplotlib.pyplot as plt
import numpy as np



def plot_tokenizers_vs_time(f1_scores, classifier):
    plt.figure()
    plt.bar(LABELS, f1_scores, width=0.3)
    plt.title("Weighted F1-score per Vectorizer for '{}' classifier".format(classifier))
    plt.ylabel("F1-score")
    # plt.yscale('log')
    plt.show()

def plot_tokenizers_vs_f1_all_in_one():
    tokenizers = ("TfIdfVectorizer", "TfIdfVectorizer (Lemmatization)", "TfIdfVectorizer (Stemming)")

    logistic_regression_f1s = (0.5337, 0.5303, 0.5314)
    linear_svm_f1s = (0.5128, 0.5074, 0.5008)
    naive_bayes_f1s = (0.4125, 0.3899, 0.3829)

    ind = np.arange(len(tokenizers))  # the x locations for the groups
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(8,6))
    rects1 = ax.bar(ind - width * 2 / 6, logistic_regression_f1s, width / 3, label="Logistic Regression")
    rects2 = ax.bar(ind, linear_svm_f1s, width / 3, label="Linear SVM")
    rects3 = ax.bar(ind + width * 2 / 6, naive_bayes_f1s, width / 3, label="Naive Bayes")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("F1-score")
    plt.title("Weighted F1-score per Vectorizer")
    plt.ylim([0, 0.7])
    ax.set_xticks(ind)
    ax.set_xticklabels(tokenizers)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    plot_tokenizers_vs_f1_all_in_one()
