"""
Project: Coin Detector in Thai Baht
Subject: Digital Image Processing 
Members:
    5710500208 Nuttapol Laotichareon
    5710501565 Nuttapon Thanitsukkan
"""

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2

class CoinDetector():
    def __init__(self, path):
        self.imgtemp = cv2.imread(path)
        self.num_coins = 0
        if not self.imgtemp is None:
            self.segment()
            cv2.waitKey(0)

    def resize_img(self, ratio):
        image = cv2.resize(self.imgtemp, None, fx=ratio, fy=ratio)
        cv2.imshow("Input", image)
        return image

    def PMSF(self, image, spatial_win_r, color_win_r):
        p_shifted = cv2.pyrMeanShiftFiltering(image, spatial_win_r, color_win_r)
        return p_shifted

    def otsu_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_val, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("Thresh", thresh_img)
        return gray, thresh_img

    def looping_watershed(self, image, gray, labels):
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cont_img, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            '''
            TODO: classify each coins
            '''
            cv2.putText(image, "{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    def my_watershed(self, image, gray, thresh_img, min_distance):
        distance_map = ndimage.distance_transform_edt(thresh_img)
        local_max= peak_local_max(distance_map, indices=False, min_distance=min_distance, labels=thresh_img)
        markers, features = ndimage.label(local_max, structure=np.ones((3, 3)), output=None)
        labels = watershed(-1 * distance_map, markers, mask=thresh_img)
        self.num_coins = len(np.unique(labels)) - 1
        image = self.looping_watershed(image, gray, labels)
        cv2.imshow("Output", image)
        return image

    def segment(self):
        image = self.resize_img(0.3)
        p_shifted = self.PMSF(image, 21, 51)
        gray, thresh_img = self.otsu_threshold(p_shifted)
        out_img = self.my_watershed(image, gray, thresh_img, 20) # min_distance is changable

    def calculate(self):
        print 'fuck i don\'t know'

    def count(self, currency='th'):
        if currency == 'th':
            print '10 Baht'
        else:
            print self.calculate()

if __name__ == '__main__':
    coins = CoinDetector(path='images/coin2.jpg')
    print coins.count(currency='th')

