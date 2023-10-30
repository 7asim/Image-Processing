import cv2
import numpy as np

def hist_ncc(image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    hist_image1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    hist_image2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

    hist_image1 /= hist_image1.sum()
    hist_image2 /= hist_image2.sum()

    hist_intersection = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_INTERSECT)
    hist_correlation = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CORREL)
    hist_chi_square = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CHISQR)
    hist_bhattacharyya = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_BHATTACHARYYA)

    ncc = np.sum((gray_image1 - np.mean(gray_image1)) * (gray_image2 - np.mean(gray_image2)) / (
                (np.std(gray_image1) * np.std(gray_image2)) * gray_image1.size))

    comparison_values = [hist_intersection, hist_correlation, hist_chi_square, hist_bhattacharyya, ncc]
    final_comparison_value = sum(comparison_values) / len(comparison_values)
    return final_comparison_value

image_path1 = r'C:\Users\asims\Desktop\Image Processing\Ball_007_0000.jpg'
image_path_comp = r'C:\Users\asims\Desktop\Image Processing\Normal_4140.jpg'

result = hist_ncc(image_path1, image_path_comp)

print(f"Comparison result: {result}")
