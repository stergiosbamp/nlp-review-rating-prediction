import nltk

from nltk.corpus import stopwords


class Preprocess:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

    def tokenize(self, sentence, keep_stopwords=False):
        """
        Method that tokenizes a sentence using NLTK's recommended word tokenizer (TreebankWordTokenizer
        along with PunktSentenceTokenizer)

        Args:
            sentence (str): The sentence to tokenize
            keep_stopwords (boolean): Whether to keep stopwords in tokenizing

        Returns:
            list: The list of the tokens from the sentence
        """
        tokens = nltk.word_tokenize(sentence)

        if keep_stopwords:
            filtered_tokens = []
            for token in tokens:
                if token not in set(stopwords.words('english')):
                    filtered_tokens.append(token)
            return filtered_tokens

        return tokens

    def stem(self, tokens):
        """
        Method that performs stemming in a list of tokens

        Args:
            tokens (list): The list of tokens to stem

        Returns:
            list: The stemmed tokens
        """

        stemmer = nltk.stem.PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
