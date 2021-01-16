import numpy as np
import matplotlib.pyplot as plt

models = ("Logistic Regression", "Linear SVM", "Naive Bayes")

unbalanced_f1s = (0.5337, 0.5129, 0.4125)
balanced_f1s = (0.5442, 0.5264, 0.5028)

ind = np.arange(len(models))  # the x locations for the groups
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width / 4, unbalanced_f1s, width / 2, label="Unbalanced data")
rects2 = ax.bar(ind + width / 4, balanced_f1s, width / 2, label="Balanced data")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-score')
ax.set_title('F1-score for unbalanced and balanced dataset per classifier')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend(loc="upper right")

plt.show()