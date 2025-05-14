import cv2
import numpy as np

def match_sift_with_boxes(pattern_gray, frame_bgr):
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(pattern_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    if des1 is None or des2 is None or len(des2) < 2:
        return frame_bgr

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = pattern_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            result = frame_bgr.copy()
            cv2.polylines(result, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)
            return result

    return frame_bgr

