import numpy as np

# Positions are counted from 0

# Read the whole file into a dictionary
def readData():
    dict = {}
    with open("iris.data", "r") as f:
        file = f.read().splitlines()
        id = 0
        for line in file:
            list = line.split(",")
            pair = [list[:-1], list[4]]
            dict[id] = pair
            id += 1
    return dict

# Access a specific object at position k on data file
def getObject(k):
    with open("iris.data", "r") as f:
        for i, line in enumerate(f):
            if i == k:
                list = line.split(",")
                pair = [list[:-1], list[4]]
                return pair

# Calculate euclidean distance
def ED(id1, id2):
    list1 = getObject(id1)[0]
    list2 = getObject(id2)[0]
    arr1 = np.array(list1, dtype=np.float32)
    arr2 = np.array(list2, dtype=np.float32)
    dist = np.linalg.norm(arr1-arr2)
    return dist

# Collection of objects
collection = readData()

# Function for performing range search
def rangeSearch(Q, r):
    result = []
    for C in collection:
        dist = ED(Q, C)
        if dist < r and C != Q:
            result.append(C)
    return result

# Function for performing lineal KNN search
def KnnSearch(Q, k):
    result = []
    for C in collection:
        if C != Q:
            dist = ED(Q, C)
            result.append([C, dist])
    result.sort(key=lambda x: x[1])
    return result[:k]

# Calculate precision: PR = #RelevantRetrievedObjects/#RetrievedObjects
def precision(Q, retrievedList):
    qClass = getObject(Q)[1]
    relevant = 0
    for objId in retrievedList:
        objClass = getObject(objId)[1]
        if qClass == objClass:
            relevant += 1
    return round(relevant/len(retrievedList), 2)


Q = [15, 82, 121]

# P1: Range Search
# The radiuses picked will allow us to obtain 7%, 37% and 52% of the objects 
# in the collection respectively, according to the distribution analysis 
# performed on R (see Distribution Analysis folder that contains R project)
radiuses = [0.5, 1.5, 2.5]
print("Range Search:")
print("-------------\n")
for q in Q:
    print("Q:", q)
    for r in radiuses:
        rs = rangeSearch(q, r)
        pr = precision(q, rs)
        print("r =", r, "-> PR =", pr)
    print()

# P2: KNN Search
K = [2, 4, 8, 16, 32]
print("KNN Search:")
print("-------------\n")
for q in Q:
    print("Q:", q)
    for k in K:
        knn = KnnSearch(q, k)
        knnObjs = [i[0] for i in knn]
        pr = precision(q, knnObjs)
        print("k =", k, "-> PR =", pr)
    print()