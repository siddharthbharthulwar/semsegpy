from main import plot_map_binary
import cv2 as cv

img1 = cv.imread(r"C:\Users\siddh\Desktop\IEEE\resultimg\1org.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread(r"C:\Users\siddh\Desktop\IEEE\resultimg\1post.png", cv.IMREAD_GRAYSCALE)

plot_map_binary(img1, img2)