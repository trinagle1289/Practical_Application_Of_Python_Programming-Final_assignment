# -*- coding: utf-8 -*-
# Programmer: 郭科顯
# Date 2022/6/15
# 在 OpenCV 裏使用背景去除
# Python 3.9.10
# numpy 1.22.4
# opencv-python 4.6.0.66
# matplotlib 3.5.2
#

import cv2
import numpy

# 建立 MOG2 背景減法器
back_sub_MOG2 = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# 抓出遮罩中的影像
# img 圖片, mask 遮罩
def fetch_img_in_mask(img, mask):
    img = cv2.bitwise_not(img, mask=mask)
    img = cv2.bitwise_not(img, mask=mask)
    return img


# 將邊界框加入到圖片中
def add_bounding_box(img, mask):
    min_area = 750  # 偵測輪廓的最小面積

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img


# 新增輪廓到圖片中
def add_contours(img, mask):
    min_area = 750  # 偵測輪廓的最小面積

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        img = cv2.drawContours(img.copy(), cnt, -1, [0, 0, 255], 3)

    return img


# 使用 MOG2 來回傳影像中的移動物件
def get_MOG2_img(img):
    fg_mask = back_sub_MOG2.apply(img)
    img = fetch_img_in_mask(img, fg_mask)
    return img


# 取得變化後的 MOG2 影像
def get_morph_MOG2_img(img):
    fg_mask = back_sub_MOG2.apply(img)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    img = fetch_img_in_mask(img, fg_mask)
    return img


# 在 MOG2 影像中加入輪廓
def get_MOG2_img_with_contours(img):
    fg_mask = back_sub_MOG2.apply(img)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    img = add_contours(img, fg_mask)

    return img


# 在 MOG2 影像中加入邊界框
def get_MOG2_img_with_bounding_box(img):
    fg_mask = back_sub_MOG2.apply(img)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    img = add_bounding_box(img, fg_mask)

    return img


# 啟動相機
# 輸入參數可使用
# 1. get_MOG2_img
# 2. get_morph_MOG2_img
# 3. get_MOG2_img_with_contours
# 4. get_MOG2_img_with_bounding_box
# 查看其結果
def run_camera(img_process_fun):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened:
        print('Unable to open')
        exit(0)

    while True:
        ret, frame = cam.read()  # 讀取相機畫面
        processed_img = None  # 預先定義轉換後的圖片

        # 如果抓不到畫面
        if frame is None:
            break

        # 如果 img_process_fun 不為 None
        if img_process_fun is not None:
            processed_img = img_process_fun(frame)

        # 顯示圖片
        cv2.imshow('Camera View', frame)
        if img_process_fun is not None:
            cv2.imshow('Processed Image', processed_img)

        # 若按下 q 鍵則離開迴圈
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # 釋放攝影機
    cam.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_camera(get_MOG2_img)
    run_camera(get_morph_MOG2_img)
    run_camera(get_MOG2_img_with_contours)
    run_camera(get_MOG2_img_with_bounding_box)

# 參考資料:
# 背景減法器-1 https://www.twblogs.net/a/5db37a70bd9eee310da04dfb
# 背景減法器-2
# https://chtseng.wordpress.com/2018/11/03/opencv%E7%9A%84%E5%89%8D%E6%99%AF%EF%BC%8F%E8%83%8C%E6%99%AF%E5%88%86%E9%9B%A2%E6%8A%80%E8%A1%93/
# 背景減法器-2 參考程式碼
# https://github.com/ch-tseng/report_BackgroundSubtractor
# 影像遮罩使用 https://steam.oxxostudio.tw/category/python/ai/opencv-mask.html
# 將函式當作其他函式的變數 https://www.geeksforgeeks.org/passing-function-as-an-argument-in-python/
# OpenCV 串流影像處理 https://blog.gtwang.org/programming/opencv-webcam-video-capture-and-file-write-tutorial/
# opencv学习笔记十：使用cv2.morphologyEx()实现开运算，闭运算，礼帽与黑帽操作以及梯度运算_耐心的小黑的博客-CSDN博客_cv2.morphologyex 使用
# https://blog.csdn.net/qq_39507748/article/details/104539673
# 輪廓的取得和使用
# https://chtseng.wordpress.com/2016/12/05/opencv-contour%E8%BC%AA%E5%BB%93/
# 畫矩形
# https://shengyu7697.github.io/python-opencv-rectangle/
