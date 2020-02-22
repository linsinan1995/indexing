'''
Filename: e:\indexing\lab1.py
Path: e:\indexing
Created Date: Saturday, February 22nd 2020, 2:35:04 pm
Author: linsinan1995

Copyright (c) 2020

Image Retrieval(Near Duplicate)
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans
print(cv2.__version__)


class imageQuerier:
    def __init__(self, images):
        self.images = {i : cv2.imread(images[i],1) for i in range(len(images)) }
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
        kp1, des1 = sift.detectAndCompute(gray,None)
        kp2, des2 = sift.detectAndCompute(gray2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
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
            scores[i] = self.match(index = i, factor = factor, draw = False, query = True)
        top4_similar_pic_index = sorted(scores,  key=scores.get, reverse = True)[:4]

        total_score = sum([scores[idx] for idx in top4_similar_pic_index])
        top4_similar_pic = {idx:scores[idx]/total_score for idx in top4_similar_pic_index}
        print(top4_similar_pic)
        return top4_similar_pic

    def plot_query_result(self, top4_similar_pic, size = (20, 15)):
        plt.figure(figsize=size)
        
        grid = plt.GridSpec(2, 4)
        plt.subplot(grid[:2, :2])
        plt.imshow(cv2.cvtColor(self.query_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Entered Picture")
        for i, idx in enumerate(top4_similar_pic):
            plt.subplot(grid[i//2, 2+i%2])
            img = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title("Top {}, Score {:.2f}".format(i+1, top4_similar_pic[idx]))
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    img = imageQuerier(["data/MarsBar1.jpg", "data/MarsBar2.jpg"])
    img.display(0)
    img.sift_kp_detect(draw=True)
    img.match()
    
    data_dir = "data/VOC2007/JPEGImages/"
    # pick 30 images in this dir
    n_pics = 30
    data_paths = [data_dir + image  for _, _, images in os.walk(data_dir) for i, image in enumerate(images) if i < n_pics]
    img_querier = imageQuerier(data_paths)

    np.random.seed(250)
    idx = np.random.randint(0, len(data_paths))
    print(idx)
    img_affined = cv2.imread(data_paths[idx], 1)

    M = np.float32([[1,0,100],[0,1,50]])
    rows, cols, _ = img_affined.shape
    img_affined = cv2.warpAffine(img_affined, M, (cols,rows))
    img_querier.display(img=img_affined)

    most_similar = img_querier.query(img_affined)
    img_querier.plot_query_result(most_similar)
