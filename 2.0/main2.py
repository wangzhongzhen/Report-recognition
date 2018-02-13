# -*- coding: utf-8 -*-
# 颜色特征+边缘特征
import cv2
import numpy as np

class Detect(object):
    def __init__(self,path):
        self.path = path
    def load_gray(self):
        img = cv2.imread(self.path)
        self.img = img
        self.colorImage = img.copy
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        cv2.imshow("hsv",hsv)
        cv2.waitKey(0)
        return hsv
    def bgr_hsv(self):
        w = np.uint8([[[255,255,255]]])
        hsv_w = cv2.cvtColor(w,cv2.COLOR_BGR2HSV)
        print(hsv_w)
        return hsv_w
    def color_area(self,W,hsv):
        if (W == True):
            lower = np.array([0,0,100])
            upper = np.array([180,30,255])
            # 根据阈值构建mask
            # 在lower和upper之间取255，其余取0
            mask = cv2.inRange(hsv,lower,upper)
            img_and = cv2.bitwise_and(hsv,hsv,mask=mask) #
            # gray = cv2.cvtColor(img_and,cv2.COLOR_BAYER_BG2GRAY)
            cv2.imshow("w", img_and)
            cv2.waitKey(0)
            return img_and

    def preprocess(self,hsv):
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(sobel_y,cv2.CV_8U,1,0,ksize=3)
        sobel = cv2.subtract(sobel_y, sobel_x)
        sobel = cv2.convertScaleAbs(sobel)

        mean_sobel = cv2.blur(sobel,ksize=(3,3))    # 消除高频噪声
        # hor = np.concatenate((sobel_y, sobel_x, sobel,mean_sobel), axis=1)
        # cv2.imshow("pre", hor)
        # cv2.waitKey(0)
        # otsu 阈值选择
        _, binary = cv2.threshold(mean_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)   # 取y轴梯度
        # 形态学处理
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(12,10))    # 第一步膨胀
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 8))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        # 膨胀
        dilation0 = cv2.dilate(binary,kernel1,iterations=5)
        # 腐蚀
        erosion0 = cv2.erode(dilation0,kernel2,iterations=14)
        # 膨胀
        image = cv2.dilate(erosion0,kernel3,iterations=11)  #
        #
        hor = np.concatenate((binary,dilation0,erosion0,image), axis=1)
        cv2.imshow("pre", hor)
        cv2.waitKey(0)

        return image

    def find_region(self,image):
        region = []
        _,contour,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        con = sorted(contour,key=cv2.contourArea,reverse=True)[0]
        rect = cv2.minAreaRect(con)  # 中心点、四个顶点、旋转角度

        box = cv2.boxPoints(rect)
        box = box.astype(int)
        return box

    # 绘图
    def draw(self,img,box):
        print(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.imshow("final_img",img)
        cv2.waitKey(0)

        cv2.imwrite("Report.jpg",img)
        return img
    def main(self):
        hsv = self.load_gray()
        # self.bgr_hsv()
        hsv_area = self.color_area(True,hsv)
        image = self.preprocess(hsv_area)
        box = self.find_region(image)
        final_img = self.draw(self.img,box)

if __name__ == "__main__":
    path = "5.jpg"        # 3
    M = Detect(path)
    M.main()














