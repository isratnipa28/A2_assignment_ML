from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib as plt
import mlflow

class LinearRegression(object):
    #cross-validation with KFold
    kfold = KFold(n_splits=3)

    def __init__(self, regularization, lr=0.001, method='batch', init_method='xavier', polynomial=True, degree=3, use_momentum=True, momentum=0.5, num_epochs=500, batch_size=50, cv=kfold):
        self.lr                 = lr
        self.num_epochs         = num_epochs
        self.batch_size         = batch_size
        self.method             = method
        self.polynomial         = polynomial
        self.degree             = degree
        self.init_method        = init_method
        self.use_momentum       = use_momentum
        self.momentum           = momentum
        self.prev_step          = 0
        self.cv                 = cv
        self.regularization     = regularization

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def r2(self, ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)
    
    def msemean(self):
        return np.sum(np.array(self.kfold_scores))/len(self.kfold_scores)
    
    def r2mean(self):
        return np.sum(np.array(self.kfold_r2))/len(self.kfold_r2)

    def fit(self, X_train, y_train):
        self.columns = X_train.columns

        if self.polynomial == True:
            X_train = self._transform_features(X_train)
            print("Using Polynomial")
        else:
            print("Using Linear")
            X_train = X_train.to_numpy()
            
        y_train = y_train.to_numpy()

        #create list of kfold scores
        self.kfold_scores = list()
        self.kfold_r2 = list()

        #reset val loss
        self.val_loss_old = np.inf
        
        #kfold.split in the sklearn
        #3 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            if self.init_method == 'xavier':
                #calculate the range for the weights with number of samples
                lower, upper = -(1 / np.sqrt(X_cross_train.shape[0])), 1 / np.sqrt(X_cross_train.shape[0])
                #randomize weights then scale them using lower and upper bounds
                numbers = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + numbers * (upper - lower)

            #initialize weights with zero
            elif self.init_method == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])

            else:
                print("Weights not initialized in init method. Must be 'xavier' or 'zero'")
                return
            
            with mlflow.start_run(run_name=f'Fold - {fold}', nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                    perm = np.random.permutation(X_cross_train.shape[0])
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]

                    #stochastic
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx].reshape(1, ) 
                            train_loss = self._train(X_method_train, y_method_train)      
                    #mini-batch
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    #batch
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)
                    
                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                    
                    yhat_val = self._predict(X_cross_val)
                    
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2_new = self.r2(y_cross_val, yhat_val)
                    
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch)

                    # Early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(val_r2_new)

                print(f"Fold {fold}: {val_loss_new}")
                print(f"Fold {fold}: {val_r2_new}")

    def _transform_features(self, X):
        X_poly = np.column_stack([X ** (self.degree)])
        return X_poly

    def _train(self, X, y):
        yhat = self._predict(X)
        m = X.shape[0]

        if self.regularization:
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        else:
            grad = (1/m) * X.T @(yhat - y)
        
        #momentum
        if self.use_momentum == True:
            self.step = self.lr * grad
            self.theta = self.theta - self.step + self.momentum * self.prev_step
            self.prev_step = self.step
        else:
            self.theta = self.theta - self.lr * grad
        
        return self.mse(y, yhat)

    def _predict(self, X):
        return X @ self.theta
    
    #predict with polynomial
    def predict(self, X):
        if self.polynomial == True:
            X = self._transform_features(X)
        return X @ self.theta
    
    def _coef(self):
        return self.theta[0:]
    
    def _bias(self):
        return self.theta[0]

    def plot_feature_importance(self, selectedfeatures):
        # Get the coefficients
        coef = self._coef()

        # If lengths don't match, truncate to the shorter one
        min_length = min(len(coef), len(selectedfeatures))
        coef = coef[:min_length]
        selectedfeatures = selectedfeatures[:min_length]

        # Create a figure and axis object with subplots
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot feature importance on the given axis
        ax.barh(selectedfeatures, coef)

        # Set labels and title for the axis
        ax.set_xlabel('Importance (Absolute Value of Coefficient)')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance')

        # Display the plot
        plt.tight_layout()
        plt.show()

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l, init_method, polynomial, degree, use_momentum, momentum):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, init_method, polynomial, degree, use_momentum, momentum)
    def msemean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l, init_method, polynomial, degree, use_momentum, momentum):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, init_method, polynomial, degree, use_momentum, momentum)
    def msemean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
    
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, l, init_method, polynomial, degree, use_momentum, momentum, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, init_method, polynomial, degree, use_momentum, momentum)
    def msemean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Normal(LinearRegression):
    
    def __init__(self, method, lr, l, init_method, polynomial, degree, use_momentum, momentum, l_ratio=0.5):
        self.regularization = None #no regularization
        super().__init__(self.regularization, lr, method, init_method, polynomial, degree, use_momentum, momentum)
    def msemean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
