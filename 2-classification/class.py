from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn import svm
import pandas as pd
import numpy as np


def make_lsi(content, titles, no_components):
    #initialixe vectorizer and transformer
    vectorizer = CountVectorizer(ENGLISH_STOP_WORDS)
    transformer = TfidfTransformer()
    #transform content and titles
    tr_content = transformer.fit_transform(vectorizer.fit_transform(content))
    tr_titles = transformer.fit_transform(vectorizer.fit_transform(titles))

    #initialize svd to use after
    truncsvd = TruncatedSVD(n_components = no_components)
    #reduce dimensions of titles and content
    reduced_titles = truncsvd.fit_transform(tr_titles)
    reduced_content = truncsvd.fit_transform(tr_content)
    #final result
    lsi_result = np.hstack((reduced_titles, reduced_content))
    return lsi_result

def find_parameters():
    pass

#get the statistics of a clasification
def get_statistics(classifier, data_lsi, data_cat, validation):

    cat_prediction = cross_val_predict(classifier, data_lsi, data_cat, cv=validation)
    accuracy = accuracy_score(data_cat, cat_prediction)
    r_score = recall_score(data_cat, cat_prediction, average='macro')
    f_score = f1_score(data_cat, cat_prediction, average='macro')
    p_score = precision_score(data_cat, cat_prediction, average='macro')

    statistics = [accuracy, p_score, r_score, f_score]
    return statistics

def find_k_nearest(k, X_train, X_test, y_train):
    pass

def nearest_neighbor_validation(k, data_lsi, data_cat, validation):
    for train_index, test_index in validation.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cat_prediction = find_k_nearest()

def main():
    #read both datasets and transform them
    train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
    test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")
    titles = list(train_data.Title)
    content = list(train_data.Content)
    test_titles = list(test_data.Title)
    test_content = list(test_data.Content)
    data_lsi = make_lsi(content, titles, 50)
    test_data_lsi = make_lsi(test_content, test_titles, 50)

    #assign to each category an int and parse all dataset categories as ints
    categories = ["Politics", "Film", "Football", "Business", "Technology"]
    cat_to_int = {categories[i] : i for i in range(len(categories))}
    data_categories = list(train_data.Category)
    #transform it directly to np array
    data_cat = np.array([cat_to_int[i] for i in data_categories])

    test_ids = list(test_data.Id)

    algorithms = [
    MultinomialNB(),
    RandomForestClassifier(n_estimators=6),
    svm.SVC(kernel='rbf', C=10, gamma=1, probability=True)] #put parameters here

    validation = KFold(n_splits=10)
    stat_array = [[] for i in range(4)]

    #start computing statistics
    #special for NB that needs a scaler for positive values
    '''scaler = MinMaxScaler(feature_range=(0, 150))
    scaled_data = scaler.fit_transform(data_lsi)
    statistics = get_statistics(algorithms[0], scaled_data, data_cat, validation)
    for j in range(4):
        stat_array[j].append(statistics[j])
    #for every other algorithm
    for i in range(1,len(algorithms)):
        statistics = get_statistics(algorithms[i], data_lsi, data_cat, validation)
        for j in range(4):
            stat_array[j].append(statistics[j])

    print stat_array'''

if __name__ == "__main__":
    main()
