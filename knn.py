import math
import numpy as np
import pandas as pd
import sklearn.metrics as SKM
import random
from collections import Counter

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    if len(a) != len(b):
        return("error, vector length does not match")
    running_sum = 0
    for ind in range(len(a)):
        running_sum += abs((a[ind]-b[ind])**2)
    # print(f"running sum: {running_sum}")
    return math.sqrt(running_sum)
    

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, k=5):

    distance_function = euclidean

    # Separate label and data
    train_labels = [item[0] for item in train]
    train_data = [item[1] for item in train]

    query_labels = [item[0] for item in query]
    query_data = [item[1] for item in query]

    predictions = []

    for query_point in query_data:
        # Calculate distances from the query point to all training points
        distances = [distance_function(query_point, train_point) for train_point in train_data]

        # Combine distances with labels
        data_with_distances = list(zip(distances, train_labels))

        # Sort the data points by distance
        data_with_distances.sort(key=lambda x: x[0])

        # Select the top k neighbors
        k_neighbors = data_with_distances[:k]

        # Count the occurrences of each label among the k neighbors
        label_counts = {}
        for _, label in k_neighbors:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Determine the most common label among the k neighbors
        most_common_label = max(label_counts, key=label_counts.get)
        predictions.append(most_common_label)

    success = sum(1 for pred, actual in zip(predictions, query_labels) if pred == actual) / len(predictions)

    return predictions



def evaluate(actual_label, predict_label, algorithm):

    '''
    Confusion matrix whose i-th row and j-th column entry indicates the number of samples
    with true label being i-th class and predicted label being j-th class.
    '''
    conMat = SKM.confusion_matrix(actual_label, predict_label)

    Recall = SKM.recall_score(actual_label, predict_label, average = None)
    RecallAvg = SKM.recall_score(actual_label, predict_label, average = "macro")

    Precision = SKM.precision_score(actual_label, predict_label, average = None, zero_division = 0.0)    
    PrecisionAvg = SKM.precision_score(actual_label, predict_label, average = "macro")

    Accuracy = SKM.accuracy_score(actual_label, predict_label)
    print(f"Accuracy with KNN:", Accuracy)
    print(f"Recall with KNN:", RecallAvg)
    print(f"Precision with KNN:", PrecisionAvg)

    with open(f"evaluate_{algorithm}.txt", "w") as f:
        labels = sorted([int(x) for x in set(actual_label)])
        f.write(f"Confusion Matrix for {algorithm}:\n")
        f.write(f"{conMat}\n")
        f.write("\n")
        f.write(f"Recall: {list(Recall)}\n")
        f.write(f"Recall Average: {RecallAvg}\n\n")
        f.write(f"Precision: {list(Precision)}\n")
        f.write(f"Precision Average: {PrecisionAvg}\n\n")
        f.write(f"Accuracy: {Accuracy}")

        
def read_data(file_name):
    
    data_set = []
    header = True
    with open(file_name,'rt') as f:
        for line in f:
            if header:
                header = False
            else:
                line = line.replace('\n','')
                tokens = line.split(',')
                label = tokens[0]
                attribs = []
                for i in range(len(tokens)-1):
                    attribs.append(float(tokens[i+1]))
                data_set.append([label,attribs])
    return(data_set)


            
def main():

    train = "Diagnostics_train.csv"
    valid = "Diagnostics_valid.csv"
    query = "Diagnostics_test.csv"
    train = read_data(train)
    valid = read_data(valid)
    query = read_data(query)

    KNN_labels = knn(train, query, k=12)
    evaluate([x[0] for x in query], KNN_labels, "KNN")

    
if __name__ == "__main__":
    main()
    
