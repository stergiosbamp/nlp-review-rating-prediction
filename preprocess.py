import nltk
import spacy

from nltk.corpus import stopwords


class Preprocess:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, document, keep_stopwords=False):
        """
        Method that tokenizes a sentence using NLTK's recommended word tokenizer (TreebankWordTokenizer
        along with PunktSentenceTokenizer)

        Args:
            document (str): The sentence to tokenize
            keep_stopwords (boolean, optional): Whether to keep stopwords in tokenizing

        Returns:
            list: The list of the tokens from the sentence
        """

        tokens = nltk.word_tokenize(document)

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

    def lemmatize(self, document, remove_punctuation=True):
        """
        Method that lemmatizes tokens based on "spaCy" library.
        Essentially internally it performs tokenization first and them finds the lemma for each token.

        Args:
            document (str): The sentence to lemmatize
            remove_punctuation (boolean, optional): Whether to keep punctuation

        Returns:
            list: The lemmatized tokens
        """

        doc = self.nlp(document)

        lemmatized_tokens = []
        for token in doc:
            if token.lemma_ != '-PRON-':
                if remove_punctuation:
                    if not (token.is_punct or token.is_space):
                        lemmatized_tokens.append(token.lemma_)
                else:
                    lemmatized_tokens.append(token.lemma_)

        return lemmatized_tokens
