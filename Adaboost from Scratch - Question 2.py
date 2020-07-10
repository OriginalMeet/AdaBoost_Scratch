# importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from mlxtend.plotting import plot_decision_regions
from sklearn.utils import shuffle
from random import randrange
from matplotlib.colors import ListedColormap
import random

################################################################### Q 2 - 1 ##############################################################
# comining two dataset's csv file to make a single dataset
dataset_q2 = pd.read_csv('classA.csv',names=['f1','f2'])
temp_dataset_q2 = pd.read_csv('classB.csv',names=['f1','f2'])
dataset_q2['class'] = pd.Series([0 for x in range(len(dataset_q2))])
temp_dataset_q2['class'] = pd.Series([1 for x in range(len(temp_dataset_q2))])
dataset_q2 = pd.concat([dataset_q2,temp_dataset_q2]).reset_index(drop=True)

# storing True False value corresponding to the class to plot the scatter plot
b = dataset_q2['class'] == 0
g = dataset_q2['class'] == 1

# plotting the class A and class B 
plt.scatter(dataset_q2['f1'][b],dataset_q2['f2'][b],c='b',label='classA')
plt.scatter(dataset_q2['f1'][g],dataset_q2['f2'][g],c='g',label='classB')
plt.title("plottig class A and class B")
plt.legend()
plt.show()

################################################################### Q 2 - 2 ##############################################################
# function to plot the decision region with decision boundary and decision margin
def plot_decision_bound_margin(best_classifier,best_C,X, y):
    # converting X and y into arrays
    X = np.asarray(X)
    y = np.asarray(y)
    # plotting scatter plot of the dataset
    plt.scatter(X[:,0], X[:,1], c=y)
    # getting min and max range for X and y
    '''Utility function to plot decision boundary and scatter plot of data'''
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Creating the meshgrid-max
    xx = np.linspace(x_min,x_max, 30)
    yy = np.linspace(y_min,y_max, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = best_classifier.decision_function(xy).reshape(XX.shape)

    # Plotting the decision margin
    plt.contour(XX, YY, Z, colors='k', levels=[-0.05, 0, 0.05], 
                        alpha=0.5, linestyles=['--', '-', '--'])
    # plotting the decision regions
    plot_decision_regions(X, y, clf=best_classifier,legend=2)
    # plotting the output plot
    plt.show()

# function for 10 times 10 fold cross validation
def cross_validation_score(dataset,differet_C, num_of_folds = 10):
    num_of_rows = dataset.shape[0] # number of rows of the dataset
    fold_size = num_of_rows / num_of_folds # calculating the fold size of the dataset
    fold_size = int(fold_size) # converting it to int
    # differnet list initialized to store different values 
    Accuracy_C = []
    Varicance_C = []
    all_the_accuracy = []
    all_the_classifiers = []
    all_the_X = []
    all_the_y = []
    # for loop for 4 different values of C
    for x in differet_C:
        classifier = []
        X_store = []
        y_store = []
        accuracy = [] # empty array for the accuacy
        for z in range(10): # 10 times
            m = 0 # m and n are used for performing 10 fold cross validation
            n = 0 
            dataset_copy = dataset
            dataset_copy = shuffle(dataset_copy).reset_index(drop=True) # shuffle the dataset
            for i in range(10): # 10 fold cross validation
                m = m + fold_size  # back limit for the cross validation split
                training_data = dataset_copy.drop(dataset_copy.index[n:m]).reset_index(drop=True) # training data
                testing_data = dataset_copy.iloc[n:m].reset_index(drop=True) # testing/ validation  data
                # splitting the dataset into X and y
                training_data_X = training_data[['f1','f2']] 
                training_data_y = training_data['class']
                testing_data_X = testing_data[['f1','f2']]
                testing_data_y = testing_data['class']
                n = n + fold_size   # front limit for the cross validation split
                # fitting the data on the Linear SVC
                svc = LinearSVC(C=x,dual=False).fit(training_data_X,training_data_y)
                # storing the accuracy obtained by the linear SVM
                linear_svc_accuracy = svc.score(testing_data_X,testing_data_y)
                classifier.append(svc)   # appending the accuracy obained
                accuracy.append(linear_svc_accuracy)
                # storing the X and y
                X_store.append(np.array(training_data_X))
                y_store.append(np.array(training_data_y))
        # stoing X, y, accuracy and classsifier
        all_the_X.append(X_store)
        all_the_y.append(y_store)
        all_the_accuracy.append(accuracy)
        all_the_classifiers.append(classifier)
        mean = np.mean(accuracy) # calculate the mean of all the accuracy
        variance = np.var(accuracy) # calculating the variance of all the accuracy
        Accuracy_C.append(mean)
        Varicance_C.append(variance)
    # obtaining the best accuracy, classifier and the dataset on which we obtained the best accuracy
    best_C_index = Accuracy_C.index(max(Accuracy_C))
    best_C_accuracy = all_the_accuracy[best_C_index]
    best_C_classifier = all_the_classifiers[best_C_index] 
    best_C_X = all_the_X[best_C_index]
    best_C_y = all_the_y[best_C_index]
    ## to plot the decision boundary of the best accuracy model
    ## let's find the model with the highest accuracy
    best_accuracy_model = np.argmax(best_C_accuracy)
    X_for_plot = best_C_X[best_accuracy_model]
    y_for_plot = best_C_y[best_accuracy_model]
    classifier_for_plot = best_C_classifier[best_accuracy_model]
    # printing the best accuracy of the model.
    print("accuracy of the best model: ",best_C_accuracy[best_accuracy_model])
    best_value_of_C = differet_C[best_C_index]
    # plotting the decision boundary and region 
    plot_decision_bound_margin(classifier_for_plot,best_value_of_C,X_for_plot,y_for_plot)
    # printing the best value of C
    print("Best value of C is: ",best_value_of_C)

    return best_value_of_C,Accuracy_C,Varicance_C

C_q2 = [0.1, 1 , 10 , 100]
# obtaining the best C
best_C,mean,variance = cross_validation_score(dataset_q2,C_q2,num_of_folds=10)
for i in range(len(C_q2)):
    print("The accuracy and mean on C = ",C_q2[i]," is ",mean[i]," and ",variance[i]," respectively")
################################################################### Q 2 - 3 ##############################################################

# function of adaboost m1 from scratch
def adaboostM1(dataset,best_C,number_of_estimator=50):
    num_of_samples = 100 # number of subsamples to be taken
    len_of_dataset = len(dataset) # length of the dataset
    count = 0 # count initialization
    # initializing different to store differnet values
    beta_stored = []
    error_stored = []
    weights_stored = []
    original_y = []
    classifier_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [],[],[],[]
    N = 100
    # initializing the weights
    weights_init = np.ones(len_of_dataset) / len_of_dataset
    # X and y for testing purposes
    X_real = dataset[['f1','f2']]
    y_real = dataset['class']

    # making a dataset with weight as a column of the dataset
    dataset_train = dataset

    # the algoritm of the adaboost using linearSVM
    while count < number_of_estimator:
        # adding a weight column to the dataset
        dataset_train['w'] = weights_init 
        dataset_copy = dataset_train
        # randomly selecting 100 rows for training purposes
        selected_rows = np.random.choice(dataset_train.index.values, 100,p=weights_init)
        data_training = dataset_train.loc[selected_rows]
        # X and y for the training purposes
        X = data_training[['f1','f2']]
        y = data_training['class']
        # fitting the data onto the linear SVM and predicting the output
        estimator = LinearSVC(C=best_C,dual=False).fit(X,y)
        y_predict = estimator.predict(X_real)
        error = 0       # initializing the error
        # For loop to calculating the error percentage using the weigths
        for a in range(len(y_predict)):
            if (y_real[a] != y_predict[a]):
                error = error + weights_init[a]
            else:
                error = error
        # if error percentage is greater then 50 % then discard the sub sample taken for training 
        if (error > 0.5):
            count = count
        else: # if error < 50 % then continue with the adaboost algorithm 
            beta = (error/(1-error))
            # For loop for reinitializing the weights
            for b in range(len(weights_init)):
                if (y_real[b] == y_predict[b]):
                    weights_init[b] = weights_init[b] * beta
                elif(y_real[b] != y_predict[b]):
                    weights_init[b] = weights_init[b] * 1
            weights_init = weights_init / np.sum(weights_init)
            # increasing the count
            count = count + 1

            beta = np.log(1/beta)
            # storing all the important information (classifier, predicted y, error, beta, weights)
            classifier_list.append(estimator)
            y_predict_list.append(y_predict)
            y = np.asarray(y)
            original_y.append(y)
            error_stored.append(error)
            beta_stored.append(beta)
            weights_stored.append(weights_init)

    y_predict_list = np.asarray(y_predict_list)
    original_y = np.asarray(original_y)
    beta_stored = np.asarray(beta_stored)
    classifier_list = np.asarray(classifier_list)
    # return the classifier list and store weight list
    return classifier_list,beta_stored

# function to predict the accuracy of the dataset
def adaboostM1_accuracy(test_data,classifiers,weights):
    len_of_dataset = len(test_data) # obtaining the length of the dataset
    # splitting the data into X and y
    X_test = test_data[['f1','f2']] 
    y_test = test_data['class']
    # initializing the list to store the classifier's predicted output
    classifiers_predicted_output = []
    # for loop to get the predicted y of all 50 classifiers
    for i in range(len(classifiers)):
        temp_output_y = classifiers[i].predict(X_test)
        classifiers_predicted_output.append(temp_output_y)
    # converting the predicted output into the array
    classifiers_predicted_output = np.asarray(classifiers_predicted_output)
    # calculting the outputing sum_of_weights to calculate the final output of 50 combined classifiers
    output_sum_of_weights = []
    for z in range(len(test_data)):
        output_sum_of_weights.append(np.sum((classifiers_predicted_output[:,z] * weights)))
    output_sum_of_weights = output_sum_of_weights / np.sum(weights)
    # initializing the list for the final y predictions
    final_predicted = []
    for x in range(len(output_sum_of_weights)):
        if (output_sum_of_weights[x] < 0.5):
            final_predicted.append(0)
        elif (output_sum_of_weights[x] > 0.5):
            final_predicted.append(1)
    # percentage accuracy of the adaboost algorithm created from scratch
    percentage_accuracy = (final_predicted == y_test).sum() / len_of_dataset
    return percentage_accuracy 

# function to plot the adaboost decsion boundary 
def plot_AdaBoost_boundary(estimators,estimator_weights, X, y, N = 1,ax = None ):
    # function to get the output from the 50 classifier of adaboost
    def AdaBoost_scratch_classify(x_temp, est,est_weights ):
        '''Return classification prediction for a given point X and a previously fitted AdaBoost'''
        temp_p = np.asarray( [ (e.predict(x_temp)).T* w for e, w in zip(est,est_weights )]  ) / est_weights.sum()
        temp = np.sum(temp_p)
        o = 0
        if (temp >= 0.5):
            o = 1
        elif (temp < 0.5):
            o = 0
        return o
    # converting X and y to the array
    X = np.asarray(X)
    y = np.asarray(y)

    # getting the min and max of X and y
    x_min, x_max = X[:, 0].min() - .001, X[:, 0].max() + .001
    y_min, y_max = X[:, 1].min() - .001, X[:, 1].max() + .001

    # creating a meshgrid based on the min and max X and y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, N),
                     np.arange(y_min, y_max, N))
    # getting the output 
    zz = np.array( [AdaBoost_scratch_classify(np.array([xi,yi]).reshape(1,-1), estimators,estimator_weights ) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
    # reshape result and plot
    Z = zz.reshape(xx.shape)
    # making color map for the contour plot
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    # plotting the final decision boundary with decision region using the meshgrid function
    plt.contourf(xx, yy, Z, 2, cmap='RdBu',alpha=0.75)
    plt.contour(xx, yy, Z,2, cmap= ListedColormap(['black','grey']))
    plt.scatter(X[:,0],X[:,1], c = y, cmap = cm_bright)
    plt.title("The decision boundary of the best adaboost classifier")
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.show()

################################################################### Q 2 - 4 & 5 ##############################################################

# function for 10 times 10 fold cross validation
def cross_validation_adaboostm1(dataset,C_best, num_of_folds = 10):
    num_of_rows = dataset.shape[0] # number of rows of the dataset
    fold_size = num_of_rows / num_of_folds # calculating the fold size of the dataset
    fold_size = int(fold_size) # converting it to int
    # initializing the list for accuracy, weights and classifiers
    all_the_accuracy = []
    all_the_weights = []
    all_the_classifiers = []
    # nested for loop for the 10 times 10 fold cross validation
    for z in range(10): # 10 times
        m = 0 # m and n are used for performing 10 fold cross validation
        n = 0 
        dataset_copy = dataset
        dataset_copy = shuffle(dataset_copy).reset_index(drop=True) # shuffle the dataset
        for i in range(10): # 10 fold cross validation
            m = m + fold_size  # back limit for the cross validation split
            training_data = dataset_copy.drop(dataset_copy.index[n:m]).reset_index(drop=True) # training data
            testing_data = dataset_copy.iloc[n:m].reset_index(drop=True) # testing/ validation  data
            n = n + fold_size   # front limit for the cross validation split
            # applying adaboost function
            classifiers, weights = adaboostM1(training_data,C_best,number_of_estimator=50)
            # getting the accuracy on the testing dataset
            scratch_adaboost_accuracy = adaboostM1_accuracy(testing_data,classifiers,weights)
            all_the_accuracy.append(scratch_adaboost_accuracy)
            all_the_classifiers.append(classifiers)
            #print("weights:",weights)
            all_the_weights.append(weights)
    # finding mean of all the accuracy
    average_accuracy = np.mean(all_the_accuracy)
    average_variance = np.var(all_the_accuracy)

    # finding the best model to plot the decision boundary
    best_accuracy_index = np.argmax(all_the_accuracy)
    best_classifier = all_the_classifiers[best_accuracy_index]
    best_weights = all_the_weights[best_accuracy_index]

    # plotting the best classifier result.
    X_plot = dataset[['f1','f2']]
    y_plot = dataset['class']
    plot_AdaBoost_boundary(best_classifier,best_weights,X_plot,y_plot)
    # returning the best accuracy and varaince
    return average_accuracy,average_variance

# getting the best accuracy and variance
accuracy,variance = cross_validation_adaboostm1(dataset_q2,best_C,num_of_folds=10)
# printing the final accuracy and variance
print("The average accuracy of adaboost M1 algorithm is: ",accuracy)
print("The variance of the adaboost M1 algorithm is: ",variance)

    
