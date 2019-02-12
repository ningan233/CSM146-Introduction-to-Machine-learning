"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        n,d = X.shape
        self.probabilities_ = np.sum(y)/n
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)

        y = None
        y = np.random.choice([0,1],n,p=[1-self.probabilities_, self.probabilities_])

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = 0
    test_error = 0
    for trial_num in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=trial_num)
        clf.fit(X_train,y_train)    # train a model using training data
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.pred(X_test)  # test the model we trained using testing features
        train_error = train_error + (1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))
		# train_error = train_error + (1 - clf.score(X_train,y_train))
        test_error = test_error + (1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))
    train_error = train_error/ntrials
    test_error = test_error/ntrials
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("../data/titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()  # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)  # fit training data using the classifier
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\ntraining error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion="entropy")  # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)  # fit training data using the classifier
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
	# train_error = 1 - clf.score(X,y)
    print('\ntraining error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
	# train_error = 1 - clf.score(X, y)
    print('\ntraining error (n_neighbors=3): %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
	# train_error = 1 - clf.score(X, y)
    print('\ntraining error (n_neighbors=5): %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
	# train_error = 1 - clf.score(X, y)
    print('\ntraining error (n_neighbors=7): %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
	
    clf1 = MajorityVoteClassifier()  # Majority Vote Classifier
    clf2 = RandomClassifier()   # Random Classifier
    clf3 = DecisionTreeClassifier(criterion="entropy")  # Decision Tree Classifier
    clf4 = KNeighborsClassifier(n_neighbors = 5)  #KNeighbors Classifier
	
    err1 = error(clf1,X,y)
    err2 = error(clf2,X,y)
    err3 = error(clf3,X,y)
    err4 = error(clf4,X,y)
	
    print('training error and testing error for MajorityVoteClassifier are: %.3f, %.3f' % (err1[0], err1[1]))
    print('training error and testing error for RandomClassifier is: %.3f, %.3f' % (err2[0], err2[1]))
    print('training error and testing error for DecisionTreeClassifier is: %.3f, %.3f' %(err3[0], err3[1]))
    print('training error and testing error for KNeighborsClassifier is: %.3f, %.3f' %(err4[0], err4[1]))
    ### ========== TODO : END ========== ###


'''
    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    from sklearn.model_selection import KFold
    K_list = []  
    Kerr_list = []  
    for i in range(25):
	    K = 2 * i + 1
        K_list.append(K) 
        kfold = KFold(n_splits=10,shuffle=False)
	    clf = KNeighborsClassifier(n_neighbors=K)
	    kf_err = 1 - cross_val_score(clf,X,y,cv=kfold)
        Kerr_list.append(kf_err)  
	    print('the validation error is: %.3f' %kf_err)
    plt.plot(K_list, Kerr_list)  
    plt.xlabel('number of neighbors K')
    plt.ylabel('validation error')
    plt.show()
	    
		
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
	height_list = []
	train_err_list = []
	test_err_list = []
    for h in range(20):
	    height = h+1
		height_list.append(height)
		clf = DecisionTreeClassifier(criterion="entropy",max_depth=height)
		err = error(clf,X,y)
		train_err_list.append(err[0])
		test_err_list.append(err[1])
	plt.plot(height_list,train_err_list)
	plt.plot(height_list,test_err_list)
    plt.xlabel('depth of decision tree')
    plt.ylabel('error')
	plt.legend('training error','testing error')
	plt.show()
	
	
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    best_height = 10
	best_k = 5
	for i in range(10):
	    fraction = 0.1*i
		tree_clf = DecisionTreeClassifier(criterion="entropy",max_depth=best_height)
		KNN_clf = KNeighborsClassifier(n_neighbors=best_k)
		tree_err[i] = error(tree_clf,X,y,100,fraction)
		KNN_err[i] = error(tree_clf,X,y,100,fraction)
    ### ========== TODO : END ========== ###


    print('Done')

'''
if __name__ == "__main__":
    main()
