from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def randomForest(X_train, y_train, X_test):

    p = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8, 20],
        'criterion' :['gini', 'entropy']
    }

    CV_clf = GridSearchCV(RandomForestClassifier(), param_grid=p)
    CV_clf.fit(X_train, y_train)
    CV_clf.predict(X_test)

    best_clf = CV_clf.best_estimator_
    best_params = CV_clf.best_params_
    y_sol = best_clf.predict(X_test)

    return best_clf, best_params, y_sol

from sklearn.ensemble import GradientBoostingClassifier

def gradientBoosting(X_train, y_train, X_test):

    p = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'learning_rate': [0.1, 0.05, 0.02, 0.01],
        'max_depth': [4, 5, 6, 7, 8],
    }

    clf = GradientBoostingClassifier(n_estimators=500)

    CV_clf = GridSearchCV(clf, param_grid=p)

    CV_clf.fit(X_train, y_train)
    CV_clf.predict(X_test)

    best_clf = CV_clf.best_estimator_
    best_params = CV_clf.best_params_
    y_sol = best_clf.predict(X_test)

    return best_clf, best_params, y_sol


from keras.layers import Input, Dense, Activation
from keras.models import Model
import numpy as np

def NN():

    np.random.seed(123+3)

    # This returns a tensor to represent the input
    x = Input(shape=(180,))
    
    # a layer instance is callable on a tensor, and returns a tensor
    h = Dense(10)(x)
    h = Activation('tanh')(h)
    h = Dense(10)(h)
    h = Activation('tanh')(h)
    h = Dense(6)(h)
    h = Activation('tanh')(h)
    h = Dense(4)(h)
    h = Activation('tanh')(h)
    h = Dense(4)(h)
    h = Activation('tanh')(h)
    h = Dense(2)(h)

    # h = Dense(10)(x)
    # h = Activation('relu')(h)
    # h = Dense(10)(h)
    # h = Activation('relu')(h)
    # h = Dense(2)(h)

    # to find out more about activations check the keras documentation
    #y = Activation('softmax')(h)
    y = Activation('sigmoid')(h)

    # Package it all up in a Model
    net = Model(x, y)

    return net



if __name__ == '__main__':

    randomForest(X_train, y_train, X_test)

    gradientBoosting(X_train, y_train, X_test)

    NN()

