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

# Coin types from largest to smallest
coinTypes = [
    1,
    2,
    5,
    10
]


class CoinDetector():
    def __init__(self, path):
        self.imgtemp = cv2.imread(path)
        self.num_coins = 0

        self.coinCount = [0, 0, 0, 0]
        self.totalPrice = 0
        if not self.imgtemp is None:
            self.segment()
            cv2.waitKey(0)

    def resize_img(self, ratio):
        image = cv2.resize(self.imgtemp, None, fx=ratio, fy=ratio)
        cv2.imshow("Input", image)
        return image

    def PMSF(self, image, spatial_win_r, color_win_r):
        p_shifted = cv2.pyrMeanShiftFiltering(
            image, spatial_win_r, color_win_r)
        return p_shifted

    def otsu_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_val, thresh_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("Thresh", thresh_img)
        return gray, thresh_img

    def looping_watershed(self, image, gray, labels):
        positions = []
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cont_img, contours, hierarchy = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            positions.append((x, y, r, label))

        largestRadius = max(positions, key=lambda x: x[2])
        smallestRadius = min(positions, key=lambda x: x[2])

        _range = (largestRadius[2] - smallestRadius[2]) / float(len(coinTypes))
        ratioPatterns = map(lambda i: smallestRadius[2] +
                            ((i + 1) * _range), range(len(coinTypes)))
        '''
           making Bias for coin 1 baht and coin 2 baht
        '''
        if len(coinTypes) > 2 and coinTypes[0] == 1 and coinTypes[1] == 2:
            ratioPatterns[0] = ratioPatterns[0] * 0.95

        for position in positions:
            (x, y, r, label) = position
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            # calculate coins :D
            coinType = 0 if r <= ratioPatterns[0] else (len(ratioPatterns) - 1)
            for i in range(1, len(ratioPatterns)):
                if r <= ratioPatterns[i] and r > ratioPatterns[i - 1]:
                    coinType = i
                    break
            self.coinCount[coinType] += 1
            self.totalPrice += coinTypes[coinType]

            cv2.putText(image, "{} Baht".format(coinTypes[coinType], r), (int(
                x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    def my_watershed(self, image, gray, thresh_img, min_distance):
        distance_map = ndimage.distance_transform_edt(thresh_img)
        local_max = peak_local_max(
            distance_map, indices=False, min_distance=min_distance, labels=thresh_img)
        markers, features = ndimage.label(
            local_max, structure=np.ones((3, 3)), output=None)
        labels = watershed(-1 * distance_map, markers, mask=thresh_img)
        self.num_coins = len(np.unique(labels)) - 1
        image = self.looping_watershed(image, gray, labels)
        print "Total Price: ", self.totalPrice
        cv2.imshow("Output", image)
        return image

    def segment(self):
        image = self.resize_img(0.3)
        p_shifted = self.PMSF(image, 21, 51)
        gray, thresh_img = self.otsu_threshold(p_shifted)
        # min_distance is changable
        out_img = self.my_watershed(image, gray, thresh_img, 20)


if __name__ == '__main__':
    print "Only coin 1, 2, 5, 10 Baht is available"
    # use set to unqiue
    coinTypes = list(set(map(int, raw_input(
        "Enter available coin (example. 1, 5, 10): ").split(","))))
    coinTypes.sort()
    coins = CoinDetector(path='images/coin11.jpg')
