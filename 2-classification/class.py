from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.neighbors import DistanceMetric
from sklearn import svm
from scipy.spatial.distance import euclidean
from operator import itemgetter
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

def find_parameters(classifier, data_lsi, data_cat): 
    Cs = [0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    kernel = ['rbf', 'linear']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}

    grid = GridSearchCV(classifier, param_grid, cv=10)
    grid.fit(data_lsi, data_cat)
    print(grid.best_params_)
    # after GridSearchCV with 2 option in kernel {'linear','rbf'} my parameters now are {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

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
    #distance = euclidean
    predictions = []
    for xts in X_test:
        dist_results = []
        #print "X_train", X_train[0]
        for j, xtr in enumerate(X_train):
            dist = euclidean(xts, xtr)
            dist_results.append((dist, y_train[j]))
        dist_results.sort(key=itemgetter(0))
        dist_results = dist_results[:k]
        #print dist_results
        #majority voting
        cat_no = [0 for i in range(5)]
        for x, i in dist_results:
            cat_no[i]+=1
        #print cat_no
        predictions.append(cat_no.index(max(cat_no)))
        #print predictions
    return predictions

def nearest_neighbor_validation(k, data_lsi, data_cat, validation):
    statistics = [[] for i in range(4)]
    mean_statistics = [0 for i in range(4)]
    for train_index, test_index in validation.split(data_lsi, data_cat):
        print "iteration" 
        X_train, X_test = data_lsi[train_index], data_lsi[test_index]
        y_train, y_test = data_cat[train_index], data_cat[test_index]
        cat_prediction = find_k_nearest(k, X_train, X_test, y_train)
        statistics[0].append(accuracy_score(y_test, cat_prediction))
        statistics[1].append(recall_score(y_test, cat_prediction, average='macro'))
        statistics[2].append(f1_score(y_test, cat_prediction, average='macro'))
        statistics[3].append(precision_score(y_test, cat_prediction, average='macro'))
    #compute mean statistics from all individual
    mean_statistics = [np.mean(i) for i in statistics]
    #print mean_statistics
    return mean_statistics


def main():
    #read both datasets and transform them
    train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
    test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")
    titles = list(train_data.Title)
    content = list(train_data.Content)
    test_titles = list(test_data.Title)
    test_content = list(test_data.Content)
    data_lsi = make_lsi(content, titles, 200)
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
    #svm.SVC()]

    #find_parameters(svm.SVC(), data_lsi, data_cat)
    
    validation = KFold(n_splits=10)
    stat_array = [[] for i in range(4)]

    #start computing statistics
    #special for NB that needs a scaler for positive values
    scaler = MinMaxScaler(feature_range=(0, 150))
    scaled_data = scaler.fit_transform(data_lsi)
    statistics = get_statistics(algorithms[0], scaled_data, data_cat, validation)
    for j in range(4):
        stat_array[j].append(statistics[j])
    #for every other algorithm
    for i in range(1,len(algorithms)):
        statistics = get_statistics(algorithms[i], data_lsi, data_cat, validation)
        for j in range(4):
            stat_array[j].append(statistics[j])

    print stat_array
    k = 3
    statistics = nearest_neighbor_validation(k, data_lsi, data_cat, validation)
    for j in range(4):
        stat_array[j].append(statistics[j])
    print statistics

    train_data_frame = pd.DataFrame(np.array(stat_array))
    train_data_frame.columns = ['Naive Bayes', 'Random Forest', 'SVM', 'KNN']
    train_data_frame.index = ['Accuracy', 'Precision', 'Recall', 'F-Measure']
    train_data_frame.to_csv("EvaluationMetric_10fold.csv", sep='\t')

    #accur_classifirer = svm.SVC(kernel='rbf', C=10, gamma=1, probability=True)
if __name__ == "__main__":
    main()
