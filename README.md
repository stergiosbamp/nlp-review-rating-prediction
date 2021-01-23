# nlp-review-rating-prediction
This is repository for the NLP course project.

## Task
Product review rating prediction is the task of predicting a review rating from 
the free-form text part of a review. It’s a relatively new machine 
learning task involving natural language processing, with potential
benefits for e-commerce, review  websites, forums, and social  media  platforms.  

We study this task as a supervised learning problem using data from Amazon’s product reviews. 
We evaluate different techniques for data pre-processing in combination with different machine learning 
models for  the  review  rating  prediction problem. We additionally investigate the use of word and 
document-level embeddings in combination with the same machine learning models.

Contributors:

* Stergios Bampakis
* Themis Spanoudis


## Dataset
The dataset utilized to train our machine learning models relies on Amazon review data. 
These can be found in:

https://nijianmo.github.io/amazon/index.html

## Project setup

Create virtual environment

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Upgrade pip

```
$ python -m pip install --upgrade pip
```

Install dependencies

```
$ pip install -r requirements.txt
```

### Running the models and experiments

Our models can be used on-the-fly with all the available dataset urls found in the link referenced above.

For example to train and evaluate a Linear SVM with the Software Amazon's category:

```
$ export PYTHONPATH=${PYTHONPATH}:$(pwd)
$ cd experiments/
$ python svm.py http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz
```

The above will download the JSON dataset in the `data` folder and also run all the experiments.
The results and the classification metrics will be automatically saved in the `findings/SVM/` directory accordingly.

### Running an embeddings experiment

To run the Doc2Vec experiment which uses the Logistic Regression classifier:

```
$ export PYTHONPATH=${PYTHONPATH}:$(pwd)
$ cd experiments/
$ python doc2vec_logistic.py http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz
```

After the first run of the experiments all the trained and pre-trained models and embeddings 
using "Gensim" are also saved in the `data/` directory for time efficiency in any subsequent runs.
