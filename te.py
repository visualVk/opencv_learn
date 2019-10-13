import unittest
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
from skimage.measure import compare_ssim


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_cv2(self):
        img = cv2.imread("lib/star.jpg")
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
        elif k == ord('s'):
            cv2.imwrite('mystar.png', img)
            cv2.destroyAllWindows()

    def test_cv2AndMatplot(self):
        img = cv2.imread("lib/star.jpg", 0)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        plt.subplot(121)
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(img2)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def test_Matplot(self):
        img = cv2.imread("lib/star.jpg", 0)
        print(img.shape)
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def test_draw_line(self):
        img = np.zeros((512, 512, 3), np.uint8)
        cv2.line(img, (0, 0), (200, 200), (255, 0, 0), 5)
        cv2.imwrite('cv2line.png', img)

    def test_draw_ellipse(self):
        img = np.zeros((512, 512, 3), np.uint8)
        cv2.ellipse(img, (100, 100), (100, 50), 20, 0, 180, (255, 0, 0), -1)
        cv2.imwrite('cv2ellipse.png', img)

    def test_itemset(self):
        img = cv2.imread('lib/star.jpg')
        print(img.item(10, 10, 2))
        img.itemset((10, 10, 2), 100)
        print(img.item(10, 10, 2))

    def test_cut(self):
        img = cv2.imread('lib/stamp1.png')
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, b, g])
        plt.subplot(131)
        plt.imshow(img2)

        stamp = img2[50:340, 60:330]
        plt.subplot(132)
        plt.imshow(stamp)

        stamp[:, :, 2] = 0
        plt.subplot(133)
        plt.imshow(stamp)
        plt.show()

    def test_merge_pic(self):
        img1 = cv2.imread(('lib/stamp1.png'))
        img2 = cv2.imread('lib/star.jpg')
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('img2gray.png', img2gray)
        ret, mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        cv2.imwrite('mask.png', mask)
        cv2.imwrite('maks_inv.png', mask_inv)

        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
        cv2.imwrite('img1_bg.png', img1_bg)
        img2_bg = cv2.bitwise_and(img2, img2, mask=mask_inv)
        cv2.imwrite('img2_bg.png', img2_bg)

        dst = cv2.add(img1_bg, img2_bg)
        img1[0:rows, 0:cols] = dst

        cv2.imshow('res', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_optimized(self):
        cv2.useOptimized()

    def test_hsv(self):
        img = cv2.imread('lib/stamp1.png')

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        upper_red = np.array([0, 255, 255])
        lower_red = np.array([0, 255, 200])
        mask = cv2.inRange(hsv_img, lower_red, upper_red)

        res_img = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(res_img)
        plt.show()

    def test_RGB2HSV(self):
        red = np.uint8([[[178, 193, 219]]])
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        print(hsv_red)

    def test_Polar2Anix(self):
        # lower_red(RGB) (208,79,113)
        # upper (212,47,56)
        r, angle = cv2.cartToPolar(1, 1)
        print(r, angle)
        x, y = cv2.polarToCart(r, angle)
        print(x, y)

    def test_extra_stamp(self):
        np.set_printoptions(threshold=np.inf)
        image = cv2.imread("lib/stamp2.png")

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_range = np.array([102, 76, 204])
        high_range = np.array([68, 54, 205])
        th = cv2.inRange(hue_image, low_range, high_range)
        cv2.imshow('th', th)
        cv2.imshow('hug_img', hue_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.imshow(th), plt.show()
        th_inv = cv2.bitwise_not(th)
        cv2.imwrite('stamp1_mask.png', th_inv)
        res_img = cv2.bitwise_and(image, image, mask=th)
        cv2.imwrite('stamp_res_img.png', res_img)
        # index1 = th_inv = 0
        # img = np.zeros(image.shape, np.uint8)
        # img[:, :] = (255, 255, 255)
        # img[index1] = image[index1]  # (0,0,255)
        # cv2.imwrite('extract_img.png',img)
        # cv2.imshow('original_img', image)
        # cv2.imshow('extract_img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 提取印章
    def test_extra_stamp2(self):
        # lower_red(RGB) (208,79,113)
        # upper (212,47,56)
        np.set_printoptions(threshold=np.inf)
        image = cv2.imread("lib/stamp4.jpg")

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_range = np.array([150, 103, 100])
        high_range = np.array([180, 255, 255])
        th = cv2.inRange(hue_image, low_range, high_range)
        # cv2.imshow('th', th)
        index1 = th == 255

        img = np.zeros(image.shape, np.uint8)
        img[:, :] = (255, 255, 255)
        img[index1] = image[index1]  # (0,0,255)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)
        cv2.imshow(dilation)
        # cv2.imwrite('stamp4_extract.png', img)
        # cv2.imshow('original_img', image)
        # cv2.imshow('extract_img', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 比较相似度
    def test_compare_pic(self):
        imageA = cv2.imread("lib/stamp2.png")
        imageB = cv2.imread("extract_img.png")

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        h1, w1 = grayA.shape
        h2, w2 = grayB.shape
        print(grayA.shape)
        print(grayB.shape)
        if grayA.shape < grayB.shape:
            grayA = cv2.resize(grayA, (w2, h2))
        else:
            grayB = cv2.resize(grayB, (w1, h1), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('grayB.png',grayB)
        cv2.imshow('grayB', grayB)
        cv2.imshow('grayA', grayA)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        score, diff = compare_ssim(grayA, grayB, full=True)
        print("SSIM: {}".format(score))

    def test_findContours(self):
        image = cv2.imread('lib/stamp2.png')
        image2Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img, contours, hierachy = cv2.findContours(image2Gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(np.size(contours))
        cv2.drawContours(img, contours, -1, (0, 0, 0), 3)
        cv2.imshow('img', img)
        # cv2.imshow('contour',contours)
        # cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_erode(self):
        blackImage = cv2.imread('lib/black_hole.png')
        kernel = np.ones((10, 10), np.uint8)
        resBlackImage1 = cv2.erode(blackImage, kernel, iterations=1)
        resBlackImage2 = cv2.erode(blackImage, kernel, iterations=2)
        resBlackImage3 = cv2.erode(blackImage, kernel, iterations=3)
        resBlackImages = np.hstack((resBlackImage1, resBlackImage2, resBlackImage3))
        cv2.imshow('resBlackImage', resBlackImages)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_dilate(self):
        stampSimple = cv2.imread('extract_img.png')
        kernel = np.ones((2, 2), np.uint8)
        resStamp1 = cv2.dilate(stampSimple, kernel, iterations=1)
        resStamp2 = cv2.erode(stampSimple, kernel, iterations=2)
        resStamp3 = cv2.erode(stampSimple, kernel, iterations=3)
        resStamps = np.hstack((stampSimple, resStamp1))
        cv2.imshow('res', resStamps)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_morph_open(self):
        stampSimple = cv2.imread('extract_img.png')
        kernel = np.ones((3, 3), np.uint8)
        resStamp = cv2.morphologyEx(stampSimple, cv2.MORPH_OPEN, kernel, iterations=1)
        results = np.hstack((stampSimple, resStamp))
        cv2.imshow('res', results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_getRedChannel(self):
        stampeSimple = cv2.imread('lib/stamp2.png')
        # org = cv2.cvtColor(stampeSimple,cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(stampeSimple)
        # bEd = cv2.add(b, r)
        # gEd = cv2.add(g, r)
        rEd = cv2.add(r, r)
        res = cv2.merge((b, g, rEd))
        # cv2.cvtColor(cv2.)
        # red = cv2.threshold(r,213,255,cv2.THRESH_BINARY)
        # print(r)
        # plt.imshow(r)
        # plt.show()
        # mask = cv2.threshold(stampeSimple,)
        cv2.imshow('red', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_gradient(self):
        circle = cv2.imread('lib/stamp2.png')
        sobelx = cv2.Sobel(circle,cv2.CV_64F,1,0,ksize=3)
        # 如果不进行绝对值，只有一边白
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(circle,cv2.CV_64F,0,1,ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        sobel = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
        cv2.imshow('sobel',sobel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_scharr(self):
        circle = cv2.imread('lib/stamp2.png')
        scharrx = cv2.Scharr(circle,cv2.CV_64F,1,0)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.Scharr(circle,cv2.CV_64F,0,1)
        scharry = cv2.convertScaleAbs(scharry)
        scharr = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
        cv2.imshow('scharr',scharr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_os_tmpfile(self):
        tempfile.m
        print(file)
if __name__ == '__main__':
    unittest.main()
