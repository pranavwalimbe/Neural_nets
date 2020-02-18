# Load the necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler
from imblearn import under_sampling, over_sampling

# Set the seed to 123
np.random.seed(123)


# load the dataset into the memory
data = pd.read_csv('Logistic_regression.csv')
data.head()
X = data.iloc[:, :-1].values
y = data.iloc[:, 20].values


# Pre-processing steps
'''You may need to clean the variables, impute the missing values and convert the categorical variables to one-hot encoded

following variables need to be converted to one_hot encoded
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

your final table should be like the following 

array(['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
       'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur',
       'job_housemaid', 'job_management', 'job_retired',
       'job_self-employed', 'job_services', 'job_student',
       'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_Basic', 'education_high.school',
       'education_illiterate', 'education_professional.course',
       'education_university.degree', 'education_unknown', 'default_no',
       'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',
       'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',
       'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
       'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',
       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
       'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',
       'poutcome_success'], dtype=object)
'''

# Your code goes here
data.isnull().any()

data.head()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,9] = labelencoder_X.fit_transform(X[:,9])
X[:,14] = labelencoder_X.fit_transform(X[:,14])





'''
separate the features and the target variable 
'''

# x = Your code goes here 
# y = Your code goes here
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7,8,9,14])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
import imblearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


'''
as your target class is imbalanced you need to use SMOTE function to balance it, first separate your data into training and testing set.
then use SMOTE function on the Training set.remember to not touch the testing set.
'''

from imblearn.over_sampling import SMOTE

# your code goes here
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(X_train, X_test)



'''
You need to eliminate variables with p-values larger than 0.05. To do so you can use the following function
# import statsmodels.api as sm

'''

# Your Code goes here




'''
Logistic Regression Model Fitting - iterative approach 
you need to complete the following functions 
'''

def sigmoid(scores):
    '''
    parameters : scores 
    does : calculate the sigmoid value of the scores
    return : the sigmoid value
    '''
    # Your code goes here
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target , weights):
    '''
    paramters : Input variables, Target variable and weights 
    does : calculate the log-likelihood 
    return : the log-likelihood
    '''

    # Your code goes here
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll
    print(ll)



def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    '''
    parameters : features, target, num_steps, learning_rate and add_intercept = False
    does : calculate the logistic regression weights, i have provided the add_intercept section of the code, to run you need to change the value from False to True 
    return : The logistic regression weights
    '''
    # Don't modify this part 
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    # Your code goes here
    weights = np.zeros(features.shape[1])
    

    ''' You need to iterate over the number of steps and update the weights in each iteration based on the gradient '''
    for step in range(num_steps):
        # Calculate the prediction value
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)


        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often - don't change this part 
        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
            weights = logistic_regression(X, y,
            num_steps = 300000, learning_rate = 5e-5, add_intercept=True)
        return weights

'''
Split the data into train and test set using 30% of the data for the test set
'''

# Your code goes here





'''
Train your logistic regression function over your training set and test it over your test set, You may need to tune the parameters to get
a better results
'''
    
# Your code goes here


import imblearn



    










