'''
Filename: e:\indexing\lab2.py
Path: e:\indexing
Created Date: Saturday, February 22nd 2020, 4:59:55 pm
Author: linsinan1995

Copyright (c) 2020 
Image Retrieval (from duplicate to categories)
'''


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class imageQuerier:
    def __init__(self, images):
        self.images = {i : cv2.resize(cv2.imread(images[i],1), (224, 224)) for i in range(len(images)) }
        self.query_image = None 
        self.size = len(images)

    def display(self, index = 0, size = (20, 15), img = None):
        if not hasattr(img, "shape"):
            img = self.images[index]
        plt.figure(figsize=size)
        plt.axis("off")
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        else:
            plt.imshow(img,cmap='gray')
        plt.show()

    def sift_kp_detect(self, index = 0, size = (20, 15), draw = False):
        image = self.images[index]
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # initialize sift key point detector
        sift = cv2.xfeatures2d.SIFT_create()
        # detect key points of input image
        kp = sift.detect(gray,None)
        if draw:
            draw_img = cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.display(None, img = draw_img)
        return kp

    def match(self, index = [0, 1], factor = 0.75, draw = True, query = False):
        if not query:
            image_one = self.images[index[0]]
            image_two = self.images[index[1]]
        else:
            image_one = self.query_image
            image_two = self.images[index]
            
        # Initiate SIFT detector
        gray = cv2.cvtColor(image_one, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image_two, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(gray, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if type(des1) == type(None) or len(des1) <= 2:
            print("Entered Picture doesnt have enough key point")
            return 0
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
        # print(cnt)
        # dict params for drawing matched picture
        if draw:
            draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), matchesMask = matchesMask, flags = 0)
            draw_image = cv2.drawMatchesKnn(image_one,kp1,image_two,kp2,matches,None,**draw_params)
            self.display(img = draw_image)
        return cnt

    def query(self, image, factor = 0.75):
        self.query_image = image
        scores = {i:0 for i in range(len(self.images))}
        for i in range(self.size):
            score = self.match(index = i, factor = factor, draw = False, query = True)
            # print("{}, {}".format(i, score))
            scores[i] = score
        top_similar_pic_index = sorted(scores,  key=scores.get, reverse = True)

        total_score = sum([scores[idx] for idx in top_similar_pic_index[:4]])
        top_similar_pic = {idx:scores[idx]/total_score for idx in top_similar_pic_index[:4]}
        
        return top_similar_pic, top_similar_pic_index

    def BOWquery(self, image):
        self.query_image = image
        # Initiate SIFT detector
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(gray, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        

    def plot_query_result(self, top_similar_pic, size = (20, 15)):
        plt.figure(figsize=size)
        
        grid = plt.GridSpec(2, 4)
        plt.subplot(grid[:2, :2])
        plt.imshow(cv2.cvtColor(self.query_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Entered Picture")
        for i, idx in enumerate(top_similar_pic):
            plt.subplot(grid[i//2, 2+i%2])
            img = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title("Top {}, Score {:.2f}".format(i+1, top_similar_pic[idx]))
            plt.axis("off")
        plt.show()

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
    iq = imageQuerier(images_train)
    idx = random_index(0, len(label_dict[14]))
    image = cv2.imread(label_dict[14][idx], 1)
    
    most_similar, similar_pic_index = iq.query(image)
    iq.plot_query_result(most_similar)
    
    cnt = 0
    for i, idx in enumerate(similar_pic_index):
        if labels_train[idx] == 14:
            cnt += 1
            if cnt % 50 == 0 or cnt == 1:
                print("Top {} similar images include {} Stop images, taking {:.4f}%".format(i+1, cnt, cnt/(i+1) * 100))
            
