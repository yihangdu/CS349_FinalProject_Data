from node import Node
import math
import collections
import csv
import sklearn.metrics as SKM

thresholds = {}

def C45(examples, default, label):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''


    def informationGain(data, feature, threshold):

        # get frequency for each class for the convenience of later calculation for parent entropy
        frequencyDict = dict(collections.Counter(classList))
        # Calculate parent entropy
        parentEntropy = 0
        for freq in frequencyDict:
          keyProbability = frequencyDict[freq] / len(classList)
          parentEntropy -= keyProbability * math.log2(keyProbability)


        childrenClass = {} # Hold split of class per value of attribute: {value1:{class1: #, class2: #,...},...}
        # Split class based on value of attribute first
        for irow in data:
            splitFeature = 0 if irow[feature] <= threshold else 1
            if splitFeature not in childrenClass:
                # value of attribute not in childrenClass
                childrenClass[splitFeature] = {irow[label]: 1}

            elif irow[label] not in childrenClass[splitFeature]:
                # class not in value of attribute
                childrenClass[splitFeature][irow[label]] = 1

            else:
                childrenClass[splitFeature][irow[label]] += 1

        # Calculate average entropy of children
        averageEntropyOfChildren = 0
        for featureValue in childrenClass:
            totalNumOfValue = sum([childrenClass[featureValue][x] for x in childrenClass[featureValue]])
            # Calculate child entropy for each value of attribute
            childEntropy = 0
            for theClass in childrenClass[featureValue]:
                classProbability = childrenClass[featureValue][theClass] / totalNumOfValue
                childEntropy -= classProbability * math.log2(classProbability)


            averageEntropyOfChildren += (totalNumOfValue / len(classList)) * childEntropy

        infoGain = parentEntropy - averageEntropyOfChildren

        return infoGain

    def mostCommonValue(data):
        values = [x[label] for x in data]
        return max(set(values), key = values.count)


    def findBestSplit(data):
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = list(data[0].keys())
        n_features.remove(label)

        for feature in n_features:
            values = sorted(set([i[feature] for i in data]))  # Unique values, sorted

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                # Split the data and calculate information gain
                gain = informationGain(data, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def splitData(data, best_feature, best_threshold):
        for irow in data:
            irow[best_feature] = 0 if irow[best_feature] <= best_threshold else 1

    # put all classes in a list for later use
    classList = [x[label] for x in examples]

    t = Node()
    t.label = mostCommonValue(examples) if classList else default # default only if root with empty data
    assert(t.label is not None)
    t.weight = len(examples)

    # All labels are the same class or attribute empty
    if len(set(classList)) == 1 or len(examples[0]) == 1:
        return t

    else:
        # Get the attribute that best classifies examples
        AStar, best_threshold = findBestSplit(examples)
        if AStar is None:
            return t
        thresholds[AStar] = best_threshold
        splitData(examples, AStar, best_threshold)

        # Create subtree for each possible value "a" for A*
        for a in [0, 1]:
            # Get subset where A* == a and remove A* from examples
            subsetOfExample = [{key: val for key, val in x.items() if key != AStar} for x in examples if x[AStar] == a]

            if subsetOfExample:
                t.label = AStar
                assert(t.label is not None)
                t.children[a] = C45(subsetOfExample, default, label)
            else: # D_a is empty, then add a leaf node with label of the most common value
                # This occurs when there are still features left but no data because
                # data is removed when subtracting data for other features
                leaf = Node()
                leaf.label = t.label
                assert(leaf.label is not None)
                leaf.weight = 0
                t.children[a] = leaf

    return t


def prune(node, examples, label):
    """
    Takes in a decision tree and a dataset for validation. Performs reduced error pruning
    to simplify the tree based on validation accuracy.
    """

    def prune_subtree(node, validation):
        if not node.children:
            return node

        #organize validation set into dict, grouping by value of node.label
        subdictset = {}
        for item in validation:
            if item[node.label] not in subdictset.keys():
                subdictset[item[node.label]] = [item]
            else:
                subdictset[item[node.label]].append(item)

        #if node child shares a key with the subdictset, recursively prune that child using the matching subdictset value as validation.
        #else, recursively prune it with no validation data
        for child in node.children.keys():
            if child in subdictset.keys():
                prune_subtree(node.children[child], subdictset[child])
            else:
                prune_subtree(node.children[child], {})

        #calculate the mode in the 'Class' attribute of the validation set. if validation is empty, prune the leaf.
        mode = None
        if len(validation) > 0:
            class_values = [item[label] for item in validation]
            counter = collections.Counter(class_values)
            mode = counter.most_common(1)[0][0]
        elif len(validation) == 0:
            node.children = {}
            node.label = mode
            return node

        #calculate the local accuracy using Counter.
        counts = collections.Counter(item[label] for item in validation)
        localct = counts[mode]

        #if the local accuracy is higher than the overall tree accuracy, prune the leaf.
        localAcc,_,_ = test(node, examples, label)
        if localct / len(validation) >= localAcc:
            node.children = {}
            node.label = mode
    prune_subtree(node, examples)  # Call the pruning function on the root node
    # print_tree(node)




def read_data(filename):
    '''
      takes a filename and returns attribute information and all the data in array of dictionaries
      '''
    # initialize variables
    out = []

    # note: you may need to add encoding="utf-8" as a parameter
    csvfile = open(filename, 'r')
    fileToRead = csv.reader(csvfile)

    headers = next(fileToRead)

    # iterate through rows of actual data
    for row in fileToRead:
        row  = [float(i) for i in row]
        row[0] = int(row[0])
        out.append(dict(zip(headers, row)))

    return out


def test(node, examples, label):
    '''
    Takes in a trained tree and a test set of examples. Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    actual_labels = []
    pred_labels = []
    failed = 0
    for x in examples:
        target = evaluate(node, x)
        pred_labels.append(target)
        actual_labels.append(x[label])
        if target != x[label]:
            failed = failed + 1

    return 1 - failed / len(examples), actual_labels, pred_labels  # accuracy



def evaluate(node, example):
    '''
    Takes in a tree and one example. Returns the Class value that the tree
    assigns to the example.
    '''
    if not node.children:
        return node.label  # Return the Class value of the leaf node
    else:
        xvalue = example[node.label]
        if node.children.get(xvalue) is not None:
            return evaluate(node.children[xvalue], example)
        else:
            return node.label

def print_tree(node, indent=""):
    """
    Recursively print the decision tree structure.
    """
    if not node.children:
        print(indent + "Class: " + str(node.label))
    else:
        print(indent + node.label)
        for value, child_tree in node.children.items():
            print(indent + "  └─ " + str(value))
            print_tree(child_tree, indent + "    ")


def main():
    train_data = read_data("../Diagnostics_train.csv")
    valid_data = read_data("../Diagnostics_valid.csv")
    test_data = read_data("../Diagnostics_test.csv")



    tree = C45(train_data, 0, "Beat")
    #print_tree(tree)
    for feature, threshold in thresholds.items():
        for row in test_data:
            row[feature] = 0 if row[feature] <= threshold else 1
        for row in valid_data:
            row[feature] = 0 if row[feature] <= threshold else 1
    prune(tree, valid_data, "Beat")
    _, actual_labels, pred_labels = test(tree, test_data, "Beat")


    print(f"Accuracy:", SKM.accuracy_score(actual_labels, pred_labels))
    print(f"Recall:", SKM.recall_score(actual_labels, pred_labels, average="macro"))
    print(f"Precision:", SKM.precision_score(actual_labels, pred_labels, average="macro"))


if __name__ == "__main__":
    main()
