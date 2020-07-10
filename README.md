# AdaBoost_Scratch

In this part you will create an adaboost classifier based on linear SVM to classify the
dataset in Question 2.
1. Load and plot ‘classA.csv’ and ‘classB.csv’ and visualize them on the same
figure.
2. Train a linear SVM with proper C value from the set {0.1, 1 , 10 , 100} and
visualize the decision boundary and report the accuracy based on
10-times-10-fold cross validation.
3. Create an ensemble of classifiers based on the Adaboost-M1 approach to classify
the dataset again. Use a linear SVM with the selected C in part 2 as your weak
learner classifier. Use T = 50 as the max number of weak learners.
Note:
I) For each iteration draw only 100 samples from the dataset to train each
classifier.
II) If the training error is higher than 50% in one iteration, discard the classifier
and re-sample the training set and train a new classifier. Continue until you have
trained 50 unique SVMs.
4. Report the mean and variance of accuracy for 10-times-10-fold cross validation
approach.
5. Visualize the decision boundary of the ensemble model on the plot in part 1.
