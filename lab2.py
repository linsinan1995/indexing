'''
Filename: e:\indexing\lab2.py
Path: e:\indexing
Created Date: Saturday, February 22nd 2020, 4:59:55 pm
Author: linsinan1995

Copyright (c) 2020 
Image Retrieval (from duplicate to categories)
'''

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tool.timer import timer
from tqdm import tqdm
from operator import is_not
from functools import partial

class imageQuerier:
    __sift = cv2.xfeatures2d.SIFT_create()

    def __init__(self, images, isDeepLearning = False):
        print("LOADING IMAGES!")
        if isDeepLearning:
            self.images = [cv2.resize(cv2.imread(images[i],0), (224, 224), cv2.INTER_LINEAR) for i in tqdm(range(len(images)))]
        else:
            self.images = [imageQuerier.__sift.detectAndCompute(cv2.imread(images[i],0), None)[1] for i in tqdm(range(len(images)))]
    
        self.paths = images
        self.query_image = None 
        self.size = len(images)
        self.flag = isDeepLearning
        self.BOW_init = False

    def match(self, index = [0, 1], factor = 0.75):
        assert(self.flag == False)
        des1 = self.query_descripor
        des2 = self.images[index]

        if type(des2) == type(None) or len(des2) <= 2:
            return 0

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
        search_params = dict(checks=50) # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]

        cnt = 0
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < factor*n.distance:
                matchesMask[i]=[1,0]
                cnt += 1
        return cnt

    def query(self, image, factor = 0.75):
        print("Query(SIFT Matcher Retrieval) Start!")
        self.query_image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.query_descripor = self.__sift.detectAndCompute(gray, None)
        
        if self.query_descripor is None or len(self.query_descripor) <= 2:
            print("Entered Picture doesnt have enough key points")
            raise AssertionError
        
        timer.start()
        scores = {i:0 for i in range(self.size)}
        for i in tqdm(range(self.size)):
            score = self.match(index = i, factor = factor)
            # print("{}, {}".format(i, score))
            scores[i] = score
        top_similar_pic_index = sorted(scores,  key=scores.get, reverse = True)

        total_score = sum([scores[idx] for idx in top_similar_pic_index[:4]])
        top_similar_pic = {idx:scores[idx]/total_score for idx in top_similar_pic_index[:4]}
        timer.end()

        return top_similar_pic, top_similar_pic_index

    def __build_vocaburary(self, K):
        list_des = list(filter(None.__ne__, self.images))
        self.vocaburary = np.vstack(list_des)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, label, self.center = cv2.kmeans(self.vocaburary, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        cnt = 0
        self.vocaburary = np.zeros((K, self.size))
        for i in range(self.size):
            if self.images[i] is None:
                continue
            n = self.images[i].shape[0]
            centered_des = label[cnt:(cnt+n)]
            for j in centered_des:
                self.vocaburary[j, i] += 1
            cnt += n

    def __descriptor_to_hist(self, descriptor_vector):
        return np.argmin(np.linalg.norm(self.center - descriptor_vector, axis=1))

    def __tfidf_weight(self, hist_query_img, n):
        tf = hist_query_img / (1+hist_query_img)
        df = (np.sum(self.vocaburary, 1) + 0.5)/ (np.sum(self.vocaburary)+0.5) 
        idf = np.log(1/df)
        return tf * idf

    def BOWquery(self, image, n = 10, K= 50):
        self.query_image = image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, query_des = self.__sift.detectAndCompute(gray, None)
        print(query_des.shape)
        if not self.BOW_init:
            print("Building Vocaburary!")
            timer.start()
            self.__build_vocaburary(K = K)
            timer.end()
            self.BOW_init = True
        
        des_to_hist = np.apply_along_axis(self.__descriptor_to_hist, 1, query_des)
        hist_query_img = np.zeros(K)
        for i in des_to_hist:
            hist_query_img[i] += 1
        
        
        weight = self.__tfidf_weight(hist_query_img, n)
        res = np.dot(hist_query_img * weight , self.vocaburary) 
        rank = np.argsort(res)[::-1]
        self.res = res
        self.hist_query_img = hist_query_img
        return rank

    def plot_query_result(self, top_similar_pic, size = (20, 15)):
        plt.figure(figsize=size)
        
        grid = plt.GridSpec(2, 4)
        plt.subplot(grid[:2, :2])
        plt.imshow(cv2.cvtColor(self.query_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Entered Picture")
        for i, idx in enumerate(top_similar_pic):
            plt.subplot(grid[i//2, 2+i%2])
            if self.flag:
                img = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(cv2.imread(self.paths[idx],1 ), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title("Top {}, Score {:.2f}".format(i+1, top_similar_pic[idx]))
            plt.axis("off")
        plt.show()

    @staticmethod
    def display(img, size = (20, 15)):
        if img is str:
            img = cv2.imread(img, 1)

        plt.figure(figsize=size)
        plt.axis("off")
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        else:
            plt.imshow(img,cmap='gray')
        plt.show()

    @staticmethod
    def plot(path, size = (20, 15)):
        img = cv2.imread(path, 1)
        imageQuerier.display(img, size)

    @classmethod
    def get_sift(cls):
        return cls.__sift

def random_index(_min, _max):
    np.random.seed(250)
    return np.random.randint(_min, _max)

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    _dict = {int(label):[] for label in directories}
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) ]
        label = int(d)
        for filename in file_names:
            images.append(filename)
            labels.append(label)
            _dict[label].append(filename)
    return images, labels, _dict

if __name__ == "__main__":
    ROOT_PATH = "data"
    train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Train")

    # load data
    images_train, labels_train, label_dict = load_data(train_data_directory)
    
    # stop => label: 14
    # sample
    iq = imageQuerier(images_train, isDeepLearning=False)
    idx = 14
    idx2 = random_index(0, len(label_dict[idx]))
    image = cv2.imread(label_dict[idx][idx2], 1)

    # top_similar_pic, top_similar_pic_index = iq.query(image)
    # iq.plot_query_result(top_similar_pic)

    rank_path = iq.BOWquery(image)

    imageQuerier.display(image)
    for p in rank_path[:20]:
        imageQuerier.plot(iq.paths[p])
    
    p = label_dict[idx][idx2]
    idx = iq.paths.index(p)
    des = iq.images[idx]

    his = np.zeros(50)
    for i in range(len(des)):
        best, loc = np.linalg.norm(des[i] - iq.center[0]), 0
        for j in range(len(iq.center)):
            dist = np.linalg.norm(des[i] - iq.center[j])
            if dist < best:
                best = dist
                loc = j
        his[loc] += 1

    def descriptor_to_hist(descriptor_vector):
        return np.argmin(np.linalg.norm(iq.center - descriptor_vector, axis=1))
    
    des_to_hist = np.apply_along_axis(descriptor_to_hist, 1, iq.images[idx])
    hist_query_img = np.zeros(50)
    for i in des_to_hist:
        hist_query_img[i] += 1

    print(his)
    print(iq.vocaburary[:,idx])
    print(iq.hist_query_img)
    print(hist_query_img)


    # imageQuerier.plot(label_dict[14][1])
    # idx = iq.paths.index(label_dict[14][1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, query_des = imageQuerier.get_sift().detectAndCompute(gray, None)
    print(query_des.shape)
    # des = iq.images[idx]
    


