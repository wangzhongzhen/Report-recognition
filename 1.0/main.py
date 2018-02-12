# -*- coding: utf-8 -*-
import cv2
import numpy as np

class Detect(object):
    def __init__(self,path):
        self.path = path
    def load_gray(self):
        gray_img = cv2.imread(self.path,cv2.IMREAD_GRAYSCALE)
        return gray_img
    def preprocess(self,img):
        # 提取边缘 对y方向求梯度
        sobel_ = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

        # otsu 阈值选择
        _,binary = cv2.threshold(sobel_,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        # horizontal0 = np.concatenate((sobel_,binary),axis=1)
        # cv2.imshow("pre", horizontal0)
        # cv2.waitKey(0)
        # 膨胀与腐蚀算子
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,5))

        dilation = cv2.dilate(binary,kernel1,iterations=6)
        # 腐蚀
        erosion = cv2.erode(dilation,kernel2,iterations=3)

        image = cv2.dilate(erosion,kernel1,iterations=1)
        self.pre_image = image
        horizontal = np.concatenate((dilation,erosion,image),axis=1)
        cv2.imshow("total_image",horizontal)
        cv2.waitKey(0)
    def find_region(self):
        region = []
        _,contour,hierarchy = cv2.findContours(self.pre_image,cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("test",_)
        # cv2.waitKey(0)
        num = len(contour)

        for i in range(num):
            cont = contour[i]
            # 去掉面积较小的轮廓
            area = cv2.contourArea(cont)
            # print(area)
            if (area<3000):
                continue

            # 找到最小矩形
            rect = cv2.minAreaRect(cont)
            # print("rect is:",rect)

            # box 是四个坐标
            box = cv2.boxPoints(rect)

            # print("box is:",box)
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])

            if (height>width*1.2):
                continue
            region.append(box)
            self.region = region

    def draw(self,img):
        for box in self.region:
            cv2.drawContours(img,[box.astype(int)],0,(0,0,255),1)

        cv2.imshow("final_img",img)
        cv2.waitKey(0)


if __name__ == "__main__":
    path = "5.jpg"
    M = Detect(path)
    # Load image
    img = M.load_gray()
    M.preprocess(img)
    M.find_region()
    RGB_img = cv2.imread(path)
    M.draw(RGB_img)

















