# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        Phi = X
        m = self.m_

        Phi = np.array([[1]] * n)
        for i in range(1, m+1):
            Phi = np.append(Phi, np.power(X, i), axis=1)
                
        ### ========== TODO : END ========== ###
        
        return Phi
    
    
    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :    #train a model using gradient descent method
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")
        
        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        
        # GD loop
        for t in xrange(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                # eta = 0.01 # change this line (now eta=0.01 by default)
                eta = float(1)/float(t+1)
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###
                
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            #self.coef_ = self.coef_ - eta*(np.dot((np.dot(X,self.coef_)-y).T,X)).T

            
            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.dot(X,self.coef_) # change this line
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)   
            
            #self.coef_ = self.coef_ - 2*eta*(np.dot(np.dot(self.coef_,X.T),X)-np.dot(X.T,y))
            self.coef_ = self.coef_ - 2*eta*np.dot(X.T,(y_pred-y).T)
            
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec
        
        print 'number of iterations: %d' % (t+1)
        
        return self
    
    
    def fit(self, X, y, l2regularize = None ) :    #train a model using closed-form
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
        """
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        self.coef_ = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)

        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        y = None
        y = np.dot(X,self.coef_)
        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        cost = 0
        cost = np.sum(np.square(y-self.predict(X)))  
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        error = 0
        error = np.sqrt(float(self.cost(X,y))/float(X.shape[0]))
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    
    
    
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    plot_data(train_data.X, train_data.y)
    plot_data(test_data.X, test_data.y)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    
    """ For testing: part d
    m = 1
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    a = model.cost(train_data.X, train_data.y)
    print(a)
    """
    m = 1
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    # part d      train a model using gradient descent method with different eta
       # eta = 0.01
    model.fit_GD(train_data.X, train_data.y,eta=pow(10,-2))
    print("when eta=0.01, W = %s" %model.coef_)
    print("the cost for GD (eta=0.01) is %s" %model.cost(train_data.X, train_data.y))
       # eta = 0.0001
    model.fit_GD(train_data.X, train_data.y,eta=pow(10,-4))
    print("when eta=0.0001, W = %s" %model.coef_)
    print("the cost for GD (eta=0.0001) is %s" %model.cost(train_data.X, train_data.y))
       # eta = 0.001
    model.fit_GD(train_data.X, train_data.y,eta=pow(10,-3))
    print("when eta=0.001, W = %s" %model.coef_)
    print("the cost for GD (eta=0.001) is %s" %model.cost(train_data.X, train_data.y))
       # eta = 0.0407
    model.fit_GD(train_data.X, train_data.y,eta=0.0407)
    print("when eta=0.0407, W = %s" %model.coef_)
    print("the cost for GD (eta=0.0407) is %s" %model.cost(train_data.X, train_data.y))
    # part f      train a model using gradient descent method with eta=1/(1+k)
    model.fit_GD(train_data.X, train_data.y)
    print("when eta=1/(1+k), W = %s" %model.coef_)
    print("the cost for GD (eta_k=1/(1+k)) is %s" %model.cost(train_data.X, train_data.y))
    # part e      train a model using closed form method
    model.fit(train_data.X, train_data.y)
    print("when using closed form solution, W = %s" %model.coef_)
    print("the cost for closed-form solution is %s" %model.cost(train_data.X, train_data.y))
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'
    train_rmse_list = []
    test_rmse_list = []
    complexity_list = []
    for m in range(11):
	    model = PolynomialRegression(m)
	    model.fit(train_data.X, train_data.y)
	    #c = model.cost(train_data.X, train_data.y)
	    #print(c)
	    train_rmse = model.rms_error(train_data.X, train_data.y)
	    
	    test_rmse = model.rms_error(test_data.X, test_data.y)
	    train_rmse_list.append(train_rmse)
	    test_rmse_list.append(test_rmse)
	    complexity_list.append(m)
    plt.plot(complexity_list,train_rmse_list)
    plt.plot(complexity_list,test_rmse_list)
    label=['training error','testing error']
    plt.xlabel('complexity')
    plt.ylabel('error')
    plt.legend(label)
    plt.show()
		
    ### ========== TODO : END ========== ###
    
    
    print "Done!"

if __name__ == "__main__" :
    main()
