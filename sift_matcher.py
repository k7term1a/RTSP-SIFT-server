import cv2
import numpy as np

def match_sift_with_boxes(pattern_gray, frame_bgr, draw_keypoints=False):
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 更精細的 SIFT 設定
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(pattern_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    result = frame_bgr.copy()

    if des1 is None or des2 is None or len(des2) < 2:
        return result

    # 比對 + Ratio Test（更嚴格）
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.6 * n.distance]

    # 畫比對成功的 Keypoints（可選）
    if draw_keypoints and good:
        matched_kp = [kp2[m.trainIdx] for m in good]
        result = cv2.drawKeypoints(result, matched_kp, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                   color=(0, 255, 0))

    # 足夠配對點再估 Homography
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is not None:
            # 把 pattern 的四個角點加上齊次座標
            h, w = pattern_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]])  # shape (4, 2)

            # 進行 affine transform（不支援 perspectiveTransform）
            dst = cv2.transform(np.array([pts]), M)  # shape (1, 4, 2)

            area = cv2.contourArea(dst)
            if area > 1000:
                cv2.polylines(result, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)
    return result
