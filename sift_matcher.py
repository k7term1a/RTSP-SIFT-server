# sift_matcher.py
import cv2
import numpy as np

def match_sift_with_boxes(pattern_img, frame_img):
    gray1 = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des2) < 2:
        return frame_img, False, 0

    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 找到比對點後畫框
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 單應矩陣
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = pattern_img.shape[:2]
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            result = frame_img.copy()
            cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            return result, True, len(good)

    return frame_img, False, len(good)
