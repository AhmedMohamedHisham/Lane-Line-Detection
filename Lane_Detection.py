import cv2
import numpy as np
import math

offset = 0


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lane(img, line1, line2):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    xp1 = 0
    yp1 = 0
    xp2 = 0
    yp2 = 0
    longline1 = np.empty(len(line1))
    longline2 = np.empty(len(line2))
    c = 0
    for line in line1:
        for x1, y1, x2, y2 in line:
            longline1[c] = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
            c = c + 1
    index1 = np.argmax(longline1)
    c = 0
    for line in line2:
        for x1, y1, x2, y2 in line:
            longline2[c] = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
            c = c + 1
    index2 = np.argmax(longline2)

    for x1, y1, x2, y2 in line1[index1]:
        y1 = img.shape[0] - y1
        y2 = img.shape[0] - y2
        m = (y2 - y1) / (x2 - x1)
        yy1 = img.shape[0] - 679
        xx1 = (yy1 - y1 + (m * x1)) / m
        y1 = img.shape[0] - y1
        y2 = img.shape[0] - y2
        yy1 = img.shape[0] - yy1
        xp1 = int(xx1)
        yp1 = yy1
        # cv2.line(blank_image, (x2, y2), (int(xx1), yy1), (255, 0, 0), thickness=5)

    for x1, y1, x2, y2 in line2[index2]:
        y1 = img.shape[0] - y1
        y2 = img.shape[0] - y2
        m = (y2 - y1) / (x2 - x1)
        yy2 = img.shape[0] - 679
        xx2 = (yy2 - y1 + (m * x1)) / m

        y1 = img.shape[0] - y1
        y2 = img.shape[0] - y2
        yy2 = img.shape[0] - yy2
        # print(xx, yy)
        xp2 = int(xx2)
        yp2 = yy2
        # cv2.line(blank_image, (x1, y1), (int(xx2), yy2), (0, 255, 0), thickness=5)

    points = np.array([[[xp1, yp1], [line1[index1][0][2], line1[index1][0][3]],
                        [line2[index2][0][0], line2[index2][0][1]], [xp2, yp2]]],
                      np.int32)
    cv2.fillPoly(blank_image, [points], (0, 255, 0))

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    lane_center = ((xp2 - xp1) / 2) + xp1
    image_xcenter = img.shape[1] / 2
    offset = (image_xcenter - lane_center) * 3.6 / img.shape[1]
    locx = 21
    locy = 25
    if offset > 0:
        text = "Vehicle is % 2.3f m right of lane center" % offset
        cv2.putText(img, text, (locx, locy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    elif offset < 0:
        text = "Vehicle is % 2.3f m left of lane center" % (offset * -1)
        cv2.putText(img, text, (locx, locy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    else:
        text = "Vehicle is aligned with the lane center"
        cv2.putText(img, text, (locx, locy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    return img


def process(image):
    # image = cv2.imread('../Project_data/test_images/straight_lines1.jpg')  # Blue  H 0.667 -->120,S 1-->255,V 1--> 255
    cv2.imshow("Image", image)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 3, height / 1.7),
        (2 * width / 3, height / 1.7),
        (width, height)]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Green H 0.33  -->60    , S 1-->255 , V 1--> 255
    # cv2.imshow("Image", image)
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)  # Red   H 0     -->0     , S 1-->255 , V 1--> 255
    # cv2.imshow("HSV", hsv_image)
    # print(hsv_image[0, 0, :])
    lower_color1 = np.array([20, 100, 200])     # range of yellow color
    upper_color1 = np.array([25, 255, 255])

    lower_color2 = np.array([0, 0, 230])        # range of white color
    upper_color2 = np.array([255, 25, 255])

    mask1 = cv2.inRange(hsv_image, lower_color1, upper_color1)
    result1 = cv2.bitwise_and(image, image, mask=mask1)

    mask2 = cv2.inRange(hsv_image, lower_color2, upper_color2)
    result2 = cv2.bitwise_and(image, image, mask=mask2)

    # result3 = cv2.bitwise_or(result1, result2)

    gray_image1 = cv2.cvtColor(result1, cv2.COLOR_RGB2GRAY)
    canny_image1 = cv2.Canny(gray_image1, 100, 120)
    # cv2.imshow("Canny", canny_image)
    cropped_image1 = region_of_interest(canny_image1,
                                        np.array([region_of_interest_vertices], np.int32), )
    # cv2.imshow("Cropped", cropped_image1)
    line1 = cv2.HoughLinesP(cropped_image1,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    gray_image2 = cv2.cvtColor(result2, cv2.COLOR_RGB2GRAY)
    canny_image2 = cv2.Canny(gray_image2, 100, 120)
    # cv2.imshow("Canny", canny_image)
    cropped_image2 = region_of_interest(canny_image2,
                                        np.array([region_of_interest_vertices], np.int32), )
    # cv2.imshow("Cropped2", cropped_image2)
    line2 = cv2.HoughLinesP(cropped_image2,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    if np.any(line1) and np.any(line2):
        image_with_lines = draw_lane(image, line1, line2)
    else:
        image_with_lines = image
    # print(offset)

    return image_with_lines


while 1:
    str1 = input('Enter P for "Photo detection" or V for "Video detection" or X to exit')
    if str1 == 'P' or str1 == 'p':
        image = cv2.imread('Project_data/test_images/straight_lines1.jpg')
        frame = process(image)
        cv2.imshow('Lane Detection of Photo', frame)
        cv2.waitKey(0)
    elif str1 == 'V' or str1 == 'v':
        cap = cv2.VideoCapture('Project_data/project_video.mp4')
        while cap.isOpened():
            success, frame = cap.read()
            frame = process(frame)
            cv2.imshow('Lane Detection of Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press "q" while the detection is running to close it
                break
        cap.release()
        cv2.destroyAllWindows()
    elif str1 == 'X' or str1 == 'x':
        break
