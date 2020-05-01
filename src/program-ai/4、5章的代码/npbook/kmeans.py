from numpy import *
import operator
import numpy as np
import pandas as pd
import math
import time



# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k, numSamples, dim):
    # numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, 2:]
        # centroid_dict[i] = dataSet[index, :]
    # return centroid_dict
    return centroids

# k-means cluster
def kmeans(dataSet, k, numSamples, dim, curr_max_cluster_no):
    centroids = initCentroids(dataSet, k, numSamples, dim)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(dataSet.shape[0]):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, 2:])
                if distance < minDist:
                    minDist = distance
                    minIndex = curr_max_cluster_no + j

            ## step 3: update its cluster
            if dataSet[i, 1] != minIndex:
                clusterChanged = True
                dataSet[i, 1] = minIndex

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(dataSet[:, 1].A == curr_max_cluster_no + j)[0]]
            centroids[j, :] = mean(pointsInCluster[:, 2:], axis=0)

    if math.isnan(centroids[0, 0]):
        # 取当前j异或
        dataSet[nonzero(dataSet[:, 1].A == curr_max_cluster_no + 1)[0]][:, 1] = curr_max_cluster_no
        centroids[0, :] = centroids[1, :]
        centroids[1, 0] = float('NaN')
    elif math.isnan(centroids[1, 0]):
        print('fixed')

    print('Congratulations, cluster complete!')
    return centroids, dataSet


def update_centroid_list(cluster_list, cluster_id, tmp_centroids, curr_max_cluster_no, k):
    if bool(cluster_list) == True:
        cluster_list.pop(cluster_id, None)
        global_distance.pop(cluster_id, None)
    for i in range(k):
        if math.isnan(tmp_centroids[i, 0]):
            continue
        cluster_list[curr_max_cluster_no] = tmp_centroids[i, :]
        curr_max_cluster_no += 1



    return cluster_list, curr_max_cluster_no

def update_clusterassemnt_list(dataSet, clusterAssment):
    dataSet = np.vstack((dataSet, clusterAssment))
    #dataSet.extend(clusterAssment)
    return dataSet

def update_existing_clusters(cluster_list, inplace_cluster_id, tmp_centroids, tmp_clusterAssment, dataSet, curr_max_cluster_no, k, split_step):
    cluster_list, curr_max_cluster_no = update_centroid_list(cluster_list, inplace_cluster_id, tmp_centroids, curr_max_cluster_no, k)
    if split_step > 0:
        dataSet = update_clusterassemnt_list(dataSet, tmp_clusterAssment)
    return cluster_list, curr_max_cluster_no, dataSet


def need_to_do_bikmeans(curr_max_cluster_no, dataSet, centroids_list):
    cluster_id, cluster_distance = pick_the_biggest_distance_cluster(curr_max_cluster_no, dataSet, centroids_list)
    print('max cluster id:' + str(cluster_id) +  'max global distance' + str(cluster_distance))
    if cluster_distance > 10:
        return cluster_id
    else:
        return -1


def split_dataset(dataSet, cluster_id):
    # candidate_dataset = [row for row in dataSet if row[:, 1] == cluster_id]
    # rest_dataset = [row for row in dataSet if row[:, 1] != cluster_id]
    candidate_dataset = dataSet[nonzero(dataSet[:, 1].A == cluster_id)[0]]
    rest_dataset = dataSet[nonzero(dataSet[:, 1].A != cluster_id)[0]]
    return rest_dataset, candidate_dataset

def pick_the_biggest_distance_cluster(curr_max_cluster_no, dataSet, centroids_list):
    for cluster_id in range(curr_max_cluster_no - 1 , curr_max_cluster_no - 3, -1):
        pointsInCluster = dataSet[nonzero(dataSet[:, 1].A == cluster_id)[0]]
        for pair in pointsInCluster:
            if cluster_id in global_distance.keys():
                global_distance[cluster_id] += euclDistance(pair[:, 2:], centroids_list[cluster_id])
            else:
                global_distance[cluster_id] = euclDistance(pair[:, 2:], centroids_list[cluster_id])

    # max_id, max_distance = max(global_distance.iteritems(), key=operator.itemgetter(1))
    max_id, max_distance = max(global_distance.items(), key=operator.itemgetter(1))
    return max_id, max_distance




if __name__ == '__main__':
    print("step 1: load data...")
    dataSet = []
    # fileIn = open('/Users/asukapan/workspace/test/book_exp1/exp1/data/kmeans/DataSet.txt')

    # for line in fileIn.readlines():
    #     lineArr = line.strip().split(',')
    #     dataSet.append([str(lineArr[0]), 0, float(lineArr[0]), float(lineArr[1])])
        # data_cluster_dict[data_id] = 0
        # data_id += 1

    df = pd.read_csv('/Users/asukapan/workspace/test/book_exp1/exp1/data/kmeans/feed_embedding_test.csv')
    df.insert(loc=1, column='cluster_id', value=0)
    dataSet = df.values.tolist()

    ## step 2: clustering...
    print("step 2: clustering...")
    dataSet = mat(dataSet)
    centroids_list = dict()
    k = 2

    # global_distance = 1000.0;
    global_distance = dict()

    split_step = 0

    numSamples = dataSet.shape[0]

    dim = 100
    # clusterAssment = mat(zeros((numSamples, 2)))


    ## step 1: init centroids

    curr_max_cluster_no = 0
    cluster_id = 0
    while(cluster_id >= 0 and split_step < 100):
        if split_step == 0:
            tmp_centroids, tmp_clusterAssment = kmeans(dataSet, k, numSamples, dim, curr_max_cluster_no)
            centroids_list, curr_max_cluster_no, dataSet = update_existing_clusters(centroids_list, cluster_id, tmp_centroids, tmp_clusterAssment, dataSet, curr_max_cluster_no, k, split_step)
        else:

            rest_dataset, candidate_dataset = split_dataset(dataSet, cluster_id)
            tmp_centroids, tmp_clusterAssment = kmeans(candidate_dataset, k, candidate_dataset.shape[0], dim, curr_max_cluster_no)
            centroids_list, curr_max_cluster_no, dataSet = update_existing_clusters(centroids_list, cluster_id, tmp_centroids, tmp_clusterAssment, rest_dataset, curr_max_cluster_no, k, split_step)

        cluster_id = need_to_do_bikmeans(curr_max_cluster_no, dataSet, centroids_list)
        split_step += 1
    # centroids, clusterAssment = kmeans(dataSet, k)
    centroid_result = pd.DataFrame.from_dict(centroids_list)
    print(len(centroid_result.columns))