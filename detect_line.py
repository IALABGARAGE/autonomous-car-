import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur,50,150)

    return canny


def region_of_interest(img):
    height = img.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550,250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image


def display_line(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (250,0,0), 10)

    return line_image

def make_coordinates(img, line_parameters):

    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    array = np.array([x1, y1, x2, y2])
    print('Coordonnées : ', array)
    return array

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    rigth_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_average)
    rigth_line = make_coordinates(img, rigth_fit_average)
    return np.array([left_line, rigth_line])

# image = cv2.imread('img/test_image.jpg')
# lane_image = np.copy(image)
# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# rectified_line = average_slope_intercept(lane_image, lines)
# img_with_line = display_line(lane_image, rectified_line)
# merge_image = cv2.addWeighted(lane_image, 0.8,  img_with_line, 1, 1)
# plt.imshow(merge_image)
# plt.show()


cap = cv2.VideoCapture("img/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped_image = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    rectified_line = average_slope_intercept(frame, lines)
    img_with_line = display_line(frame, rectified_line)
    merge_image = cv2.addWeighted(frame, 0.8, img_with_line, 1, 1)
    cv2.imshow("result", merge_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
