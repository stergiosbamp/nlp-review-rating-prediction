import pandas as pd

import matplotlib.pyplot as plt

from extract import Extractor


if __name__ == "__main__":
    filename = "../data/Software_5.json"
    data_fields = ["reviewText"]
    target_field = "overall"
    df = Extractor.extract_examples(filename,
                                      data_fields,
                                      target_field,
                                      drop_duplicates=True)
    
    df_target = df['overall']

    counts = df_target.value_counts()
    counts.plot.pie()
    
    plt.title("Class distribution of Software product category reviews")
    plt.show()
