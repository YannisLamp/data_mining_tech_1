from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd

if __name__ == "__main__":

    train_data = pd.read_csv('datasets/train_set.csv', sep="\t")
    train_data = train_data[0:25]
    print(train_data)



