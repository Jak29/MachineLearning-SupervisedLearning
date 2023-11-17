import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import time

# Reading in the dataset
fashion = pd.read_csv("fashion-mnist_train.csv")


# Task 2: Create a k-fold cross-validation procedure
# Task 2: Parameterise the number of samples to use from the dataset to be able to control
#         the runtime of the algorithm evaluation 
def k_fold_cross_validation(classifier, X, y, samples):
    
    # Reduce the dataset size to the amount of samples chosen
    X = X[:samples]
    y = y[:samples]

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    
    # Lists for stats of each fold
    train_times = []
    prediction_times = []
    accuracies = []

    for train_index, test_index in kf.split(X):
        # Task 2: Split the data into training and evaluation subsets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Clone the classifier so I have a new untouched one for each fold
        model = clone(classifier)
        
        # Task 2: Processing time required for training
        start_time = time.time()
        # Model training
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        # Task 2: Processing time required for prediction
        start_time = time.time()
        # Model prediction
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        prediction_times.append(prediction_time)
        
        # Task 2: Determine the confusion matrix and accuracy score of the classification
        C = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Add the accuracy for this fold into a list of the accuracies for every fold on this classifier
        accuracies.append(accuracy)

    # Task 2: Calculate the minimum, the maximum, and the average of each sample for:
    # Task 2: training time
    # Task 2: prediciton time 
    # Task 2: prediciton accuracy
    stats = {
        'min_train': min(train_times),
        'max_train': max(train_times),
        'mean_train': np.mean(train_times),
        'min_prediction': min(prediction_times),
        'max_prediction': max(prediction_times),
        'mean_prediction': np.mean(prediction_times),
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'mean_accuracy': np.mean(accuracies),
        'confusion_matrix': C
    }

    return stats


# Used for Task 3,4,5,6
def train(classifier, name):
    # Prepare the feature matrix and target vector
    X = fashion.drop('label', axis=1).values
    y = fashion['label'].values
    
    # The different sample sizes
    samples = [100, 500, 1000, 3000, 5000]  
    
    # Lists that will be used to plot the relationship between input data size and runtimes for the classifier
    mean_acc = []
    training_times = []
    predicition_times = []
    
    # Train and evaluate the classifier for every sample size
    for sample in samples:
        # Print out the current sample size being used
        print("Current Sample Size: " + str(sample))
        
        # The k fold cross validation for the classifiers
        stats = k_fold_cross_validation(classifier, X, y, sample)
        
        # Print out all the stats like training time, prediction time and prediction accuracy
        # When it comes to confusion matrix just printing it on the next line because it looks better
        for key, value in stats.items():
            if key == "confusion_matrix":
                print(f"{key}: \n{value}")
            else:
                print(f"{key}: {value}")
        print("\n")
        
        
        # Adding the values for this fold to the lists to plot
        mean_acc.append(stats['mean_accuracy'])
        training_times.append(stats['mean_train'])
        predicition_times.append(stats['mean_prediction'])
    
    # Task: Plot the relationship between input data size and runtimes for the classifier
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(samples, training_times, label='Training Time', marker='o')
    plt.plot(samples, predicition_times, label='Prediction Time', marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Runtime (in seconds)')
    plt.title('Runtimes')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(samples, mean_acc, label='Mean Accuracy', marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.suptitle(name)
    plt.show()
    
    # Return the mean accuracy across all samples for the classifier
    return str(np.mean(mean_acc))


def main():
    # Task 1: Load all sandals, sneakers, and ankle boots from the dataset 
    sandals = fashion[fashion['label'] == 5]
    sneakers = fashion[fashion['label'] == 7]
    ankle_boots = fashion[fashion['label'] == 9]

    # Task 1: Separate the labels from the feature vectors
    sandals_features = sandals.drop('label', axis=1)
    sneakers_features = sneakers.drop('label', axis=1)
    ankle_boots_features = ankle_boots.drop('label', axis=1)
        
    # Task 1: Displaying an image of a sandal
    # -------------------------------
    # Reshaping it back into a 28x28 pixel image
    sandal_img = sandals_features.iloc[0].to_numpy().reshape(28,28)
    # Displaying the image and make it gray
    plt.imshow(sandal_img, cmap="gray")
    plt.show()
    
    # Task 1: Displaying an image of sneaker
    # ------------------------------
    # Reshaping it back into a 28x28 pixel image
    sneaker_img = sneakers_features.iloc[0].to_numpy().reshape(28,28)
    # Displaying the image and make it gray
    plt.imshow(sneaker_img, cmap="gray")
    plt.show()
    
    # Task 1: Displaying an image of an ankle boot
    # ------------------------------------
    # Reshaping it back into a 28x28 pixel image
    ankle_boot_img = ankle_boots_features.iloc[0].to_numpy().reshape(28,28)
    # Displaying the image and make it gray
    plt.imshow(ankle_boot_img, cmap="gray")
    plt.show()
    
    
    # Task 3: 
    perceptron_classifier = Perceptron()
    print("\nPerceptron Classifier\n====================================\n")
    print("Mean Accuracy: " + train(perceptron_classifier, "Perceptron Classifier"))
    
    
    # Task 4:
    decision_tree_classifier = DecisionTreeClassifier()
    print("\nDecision Tree Classifier\n====================================\n")
    print("Mean Accuracy: " + train(decision_tree_classifier, "Decision Tree Classifier"))
    
    
    # Task 5:
    k_nearest_neighbours_classifier1 = KNeighborsClassifier(n_neighbors=1)
    k_nearest_neighbours_classifier2 = KNeighborsClassifier(n_neighbors=2)
    k_nearest_neighbours_classifier3 = KNeighborsClassifier(n_neighbors=3)
    k_nearest_neighbours_classifier4 = KNeighborsClassifier(n_neighbors=4)
    k_nearest_neighbours_classifier5 = KNeighborsClassifier(n_neighbors=5)
    k_nearest_neighbours_classifier6 = KNeighborsClassifier(n_neighbors=6)
    k_nearest_neighbours_classifier7 = KNeighborsClassifier(n_neighbors=7)
    k_nearest_neighbours_classifier8 = KNeighborsClassifier(n_neighbors=8)
    k_nearest_neighbours_classifier9 = KNeighborsClassifier(n_neighbors=9)
    k_nearest_neighbours_classifier10 = KNeighborsClassifier(n_neighbors=10)
    
    # print("\nK Nearest Neighbours Classifier 1\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier1, "K Nearest Neighbours Classifier 1"))
    # print("\nK Nearest Neighbours Classifier 2\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier2, "K Nearest Neighbours Classifier 2"))
    # print("\nK Nearest Neighbours Classifier 3\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier3, "K Nearest Neighbours Classifier 3"))
    # print("\nK Nearest Neighbours Classifier 4\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier4, "K Nearest Neighbours Classifier 4"))
    # print("\nK Nearest Neighbours Classifier 5\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier5, "K Nearest Neighbours Classifier 5"))
    # print("\nK Nearest Neighbours Classifier 6\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier6, "K Nearest Neighbours Classifier 6"))
    # print("\nK Nearest Neighbours Classifier 7\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier7, "K Nearest Neighbours Classifier 7"))
    # The most accurate
    print("\nK Nearest Neighbours Classifier 8\n====================================\n")
    print("Mean Accuracy: " + train(k_nearest_neighbours_classifier8, "K Nearest Neighbours Classifier 8"))
    # print("\nK Nearest Neighbours Classifier 9\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier9, "K Nearest Neighbours Classifier 9"))
    # print("\nK Nearest Neighbours Classifier 10\n====================================\n")
    # print("Mean Accuracy: " + train(k_nearest_neighbours_classifier10, "K Nearest Neighbours Classifier 10"))

    
    
    # Task 6: 
    support_vector_machine_classifier1 = SVC(gamma=0.001)
    support_vector_machine_classifier2 = SVC(gamma=0.01)
    support_vector_machine_classifier3 = SVC(gamma=0.5)
    support_vector_machine_classifier4 = SVC(gamma=1)
    support_vector_machine_classifier5 = SVC(gamma=10)
    
    # The most accurate
    print("\nSVC 1\n====================================\n")
    print("Mean Accuracy: " + train(support_vector_machine_classifier1, "SVM 1"))
    # print("\nSVC 2\n====================================\n")
    # print("Mean Accuracy: " + train(support_vector_machine_classifier2, "SVM 2"))
    # print("\nSVC 3\n====================================\n")
    # print("Mean Accuracy: " + train(support_vector_machine_classifier3, "SVM 3"))
    # print("\nSVC 4\n====================================\n")
    # print("Mean Accuracy: " + train(support_vector_machine_classifier4, "SVM 4"))
    # print("\nSVC 5\n====================================\n")
    # print("Mean Accuracy: " + train(support_vector_machine_classifier5, "SVM 5"))
    
    print("\nDone")
    
main()
    