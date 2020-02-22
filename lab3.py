import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans


class imageCls:

    def __init__(self, images):
        self.images = {i : images[i] for i in range(len(images)) }
        self.query_image = None 

    def display(self, index = 0, size = (20, 15), img = None):
        if not hasattr(img, "shape"):
            img = cv2.imread(self.images[index], 1)
        plt.figure(figsize=size)
        plt.axis("off")
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        else:
            plt.imshow(img,cmap='gray')
        plt.show()

    def sift_kp_detect(self, index = 0, size = (20, 15), draw = False):
        image = cv2.imread(self.images[index], 1)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # initialize sift key point detector
        sift = cv2.xfeatures2d.SIFT_create()
        # detect key points of input image
        kp = sift.detect(gray,None)
        if draw:
            draw_img = cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.display(None, img = draw_img)
        return kp

    def match(self, index = [0, 1], factor = 0.75, img = None):
        image_one = cv2.imread(self.images[index[0]], 1)
        image_two = cv2.imread(self.images[index[1]], 1)
        gray = cv2.cvtColor(image_one, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image_two, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
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
        # dict params for drawing matched picture
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), matchesMask = matchesMask, flags = 0)
        draw_image = cv2.drawMatchesKnn(image_one,kp1,image_two,kp2,matches,None,**draw_params)
        self.display(img = draw_image)

    def query(self, image, factor = 0.75):
        pass

if __name__ == "__main__":
    img = imageCls(["MarsBar1.jpg", "MarsBar2.jpg"])
    img.display(0)
    img.sift_kp_detect(draw=True)
    img.match()
    

# args = [3, 6]
# list(range(*args))