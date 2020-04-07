import cv2
import numpy as np
#
#   Image Alignment (Pyramid Method)
#
class ImageAlignment():
    
    def __init__(self, threshold=4):
        self.thres = threshold

    def gradient_magnitude(self, I):
        r, l, t, b = np.zeros(I.shape), np.zeros(I.shape), np.zeros(I.shape), np.zeros(I.shape)
        r[:, :-1] = I[:, 1:]; r[:, -1] = I[:, -1]
        l[:, 1:] = I[:, :-1]; l[:, 0] = I[:, 0]
        t[1:] = I[:-1]; t[0] = I[0]
        b[:-1] = I[1:]; b[-1] = I[-1]
        Ix = r - l
        Iy = b - t
        return np.sqrt(Ix ** 2 + Iy ** 2).astype(np.uint8)

    def translation_matrix(self, dx, dy):
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        return M

    def find_shift(self, src, tar, x, y):
        h, w = tar.shape[:2]
        min_error = np.inf
        best_dx, best_dy = 0, 0
        Im_tar = self.gradient_magnitude(tar)
        Im_src = self.gradient_magnitude(src)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                Im_tmp_src = cv2.warpAffine(Im_src, self.translation_matrix(x + dx, y + dy), (w, h))

                error = np.sum(np.abs(np.sign(Im_tmp_src - Im_tar)))
                if error < min_error:
                    min_error = error
                    best_dx, best_dy = dx, dy

        return x + best_dx, y + best_dy

    def align(self, src, tar, depth):
        if depth == 0:
            dx, dy = self.find_shift(src, tar, 0, 0)

        else:
            h, w = src.shape[:2]
            half_src = cv2.resize(src, (w//2, h//2))
            half_tar = cv2.resize(tar, (w//2, h//2))
            prev_dx, prev_dy = self.align(half_src, half_tar, depth-1)
            dx, dy = self.find_shift(src, tar, prev_dx * 2, prev_dy * 2)

        return dx, dy

    def fit(self, src, tar, depth=4):
        h, w, c = tar.shape
        dx, dy = self.align(src, tar, depth)
        shift = self.translation_matrix(dx, dy)
        return cv2.warpAffine(src, shift, (w, h))
