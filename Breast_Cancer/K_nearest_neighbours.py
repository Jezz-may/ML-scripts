import numpy as np
import warnings
from collections import Counter
import pandas as pd  # used to load in the dataset
import random  # used to shuffle the dataset
from sklearn import model_selection, neighbors


def k_nearest_neighbors(data, new_data, k=5):
    if len(data) >= k:
        warnings.warn('K is too small')
    distances = []  # creates an empty list, it will be populated with the distances between the point.
    for category in data:  # iterates through each category in the data
        for features in data[category]:  # iterates through each point in the category
            euclid_dis = np.linalg.norm(np.array(features) - np.array(new_data))  # calculating the euclidean distance from the new point to the feature point.
            distances.append([euclid_dis, category])  # populating the array with a list of arrays containing the distance from the new point and the category that that point was in.
    votes = [i[1] for i in sorted(distances)[:k]]  # sorts the array in ascending order and takes the category of the first k values.
    vote_result = Counter(votes).most_common(1)[0][0]  # calculates the most common category from votes.
    confidence = Counter(votes).most_common(1)[0][1] / k  # calculates how confident the result of the vote is.
    return vote_result, confidence


df = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
df.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
              'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']
df.replace('?', -10000, inplace=True)  # replacing all question marks with negative 10000
df.drop(['id'], 1, inplace=True)  # dropping the id column as it doesn't contain any useful information.
full_data = df.astype(float).values.tolist()  # some values have quotation marks
random.shuffle(full_data)  # shuffling the data

test_size = 0.15
train_dict = {2: [], 4: []}  # creating a training dictionary
test_dict = {2: [], 4: []}  # creating a testing dictionary
train_data = full_data[:-int(test_size*len(full_data))]  # splitting the data into testing and training
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_dict[i[-1]].append(i[:-1])  # populating the training dict
for i in test_data:
    test_dict[i[-1]].append(i[:-1])  # populating the testing dict

correct = 0
total = 0
for group in test_dict:  # iterates through both keys
    for datapoint in test_dict[group]:  # iterates through each set of values in the key
        vote_result, confidence = k_nearest_neighbors(train_dict, datapoint, k=5)
        if group == vote_result:
            correct += 1  # adding 1 to the correct score
        #else: print(confidence)  # prints the confidence for a data point incorrectly classified.
        total += 1
accuracy_scratch = (correct/total)*100
print(accuracy_scratch, '%')

########################################################################################################################

# using scikit-learn KNN. The default number of neighbours is 5.
X = np.array(df.drop(['class'], 1))  # dropping the label from the features
y = np.array(df['class'])  # defining the label
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)  # splitting the data into training and testing
algo = neighbors.KNeighborsClassifier()  # defining the classifier
algo.fit(X_train, y_train)  # fitting the classifier to the training data
accuracy_sklearn = algo.score(X_test, y_test)  # applying the algorithm to the test data
print(accuracy_sklearn, '%')

