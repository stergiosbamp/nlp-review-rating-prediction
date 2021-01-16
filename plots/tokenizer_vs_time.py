import matplotlib.pyplot as plt
import numpy as np



def plot_tokenizers_vs_time(vectorizers_time, classifier):
    plt.figure()
    plt.bar(LABELS, vectorizers_time, width=0.3)
    plt.title("Train and test time per Vectorizer for '{}' classifier".format(classifier))
    plt.ylabel("Train and test time (secs)")
    plt.yscale('log')
    plt.show()

def plot_tokenizers_vs_time_all_in_one():
    tokenizers = ("TfIdfVectorizer", "TfIdfVectorizer (Lemmatization)", "TfIdfVectorizer (Stemming)")

    logistic_regression_times = (1.7131, 35.9588, 226.3316)
    linear_svm_times = (2.4967, 32.4222, 225.6811)
    naive_bayes_times = (0.9424, 30.7023, 235.69)

    ind = np.arange(len(tokenizers))  # the x locations for the groups
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width * 2 / 6, logistic_regression_times, width / 3, label="Logistic Regression")
    rects2 = ax.bar(ind, linear_svm_times, width / 3, label="Linear SVM")
    rects3 = ax.bar(ind + width * 2 / 6, naive_bayes_times, width / 3, label="Naive Bayes")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Train and test time (secs)")
    plt.yscale('log')
    plt.title("Train and test time per Vectorizer")
    ax.set_xticks(ind)
    ax.set_xticklabels(tokenizers)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    plot_tokenizers_vs_time_all_in_one()
