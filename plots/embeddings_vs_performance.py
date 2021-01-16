import numpy as np
import matplotlib.pyplot as plt

models = ("TfidfVectorizer", "Mean 100-d GloVe Embeddings\n(pre-trained on Twitter data)", "Mean 100-d Word2Vec Embeddings", "100-d Doc2Vec Embeddings")

accuracies = (0.5436, 0.3998, 0.3954, 0.4530)
weighted_precisions = (0.5469, 0.4690, 0.4600, 0.4419)
weighted_recalls = (0.5436, 0.3998, 0.3954, 0.4530)
weighted_f1s = (0.5442, 0.4185, 0.4142, 0.4399)
maes = (0.6818, 1.0353, 1.0306, 1.0306)

men_means = (20, 35, 30, 35, 27)
women_means = (25, 32, 34, 20, 25)

ind = np.arange(len(models))  # the x locations for the groups
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width * 3 / 8, accuracies, width / 4, label="Accuracy")
rects2 = ax.bar(ind - width * 1 / 8, weighted_precisions, width / 4, label="Precision")
rects3 = ax.bar(ind + width * 1 / 8, weighted_recalls, width / 4, label="Recall")
rects4 = ax.bar(ind + width * 3 / 8, weighted_f1s, width / 4, label="F1")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by features and metric of LogisticRegression classifier')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend()

plt.show()
