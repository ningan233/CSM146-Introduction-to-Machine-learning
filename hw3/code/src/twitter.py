"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        
        count_key = 0   # count the number of keys in a dictionary
        for rows in fid:   #对于fid里每一行string进行loop(不需要用i来指代某一行，直接可以用内容)
            words_array = extract_words(rows) 
            for word in words_array:   #对于words_array里每一个word进行loop
                if word not in word_list:
                    word_list[word] = count_key   # key-value pair 
                    count_key += 1
        print(count_key)
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        
        for row_idx,rows in enumerate(fid):
            words_array = extract_words(rows)
            for word in words_array:
                column_idx = word_list[word]     #对于fid里面的某一行里的某一个word，找到他在word_list中的位置
                feature_matrix[row_idx][column_idx] = 1   #将该行该列元素赋值为1，表示在这一行文本中出现了这个word
            
            
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):   # use “accuracy” by default
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)     # assign y_labels by y_pred
    y_label[y_label==0] = 1       # y_pred>=0, labeled as 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    
    if metric == "accuracy" :
        model_score = metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score" :
        model_score = metrics.f1_score(y_true, y_label)
    elif metric == "auroc" :
        model_score = metrics.roc_auc_score(y_true, y_label)
        
    return model_score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold   # kf---一个train_idx,test_index的matrix 行数为k
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance    
    score_list = []
    
    for train_idx,test_idx in kf:    #通过skf得到的train_idx,test_idx是一个k行的array，每一行是一组train_data和test_data，可以用来train model和predict以及evaluate
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[test_idx]
        y_val = y[test_idx]
        clf.fit(X_train,y_train)   # train a SVM model using training data
        y_val_pred = clf.decision_function(X_val)    # predict y_val_pred for validation set using the model we trained 
        score = performance(y_val, y_val_pred, metric="accuracy")  # (这种training data和testing data label下的score)evaluate the model using validation y_pred and y_true
        score_list.append(score)
        
    avg_score = np.sum(score_list)/float(len(score_list))
    return avg_score
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    c_score_list = []
    for c in C_range:
        clf = SVC(kernel='linear', C=c) 
		#kf是这个函数的argument，所以此处不再需要写StratifiedKFold cross validation中分为training data和testing data的过程
        c_score = cv_performance(clf, X, y, kf, metric="accuracy")
        c_score_list.append(c_score)
    print c_score_list
    c_opt = C_range[np.argmax(c_score_list)] #c_opt是使score最高的那个c的值
    return c_opt
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric="accuracy") 
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    extract_dictionary('../data/tweets.txt')
    metric_list = ["accuracy", "f1_score", "auroc"]
    
    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set
    X_train = X[0:560,:]
    y_train = y[0:560]
    
    X_test = X[560:-1,:]
    y_test = y[560:-1]
    
    print("We have finished the feature extraction and generated the train/test splits!")
    
    """
    # part 2: create stratified folds (5-fold CV)
    skf = StratifiedKFold(y_train,n_folds = 5)     
         ####skf = skf.split(X_train,y_train)    #*******sklearn.model_selection和sklearn.cross_validation的区别
		 
    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    c_opt_list = []
    for metric in metric_list:
        c_opt = select_param_linear(X_train, y_train, skf, metric=metric)
        c_opt_list.append(c_opt)
        print(c_opt)
        
    # part 3: train linear-kernel SVMs with selected hyperparameters
    clf = SVM.fit(kernel='linear', C=c)
    clf.fit(X_train,y_train)                    # ********需要修改这里的loop c&metric有对应关系**********
    
    # part 3: report performance on test data
    score_list = []
    for metric in metric_list:
        score = performance_test(clf, X_test, y_test, metric=metric)
        score_list.append(score)
    ### ========== TODO : END ========== ###
    
""" 
if __name__ == "__main__" :
    main()
