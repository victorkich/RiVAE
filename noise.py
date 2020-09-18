import numpy as np
import cv2


def default(raw_img):
    return raw_img


def cartooning(raw_rgb):
    num_down_samples = 2
    num_bilateral_filters = 50
    img_color = raw_rgb
    for _ in range(num_down_samples):
        img_color = cv2.pyrDown(img_color)

    for _ in range(num_bilateral_filters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    for _ in range(num_down_samples):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3)
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    (x, y, z) = img_color.shape
    img_edge = cv2.resize(img_edge, (y, x))
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)


def gauss(raw_rgb):
    row, col, ch = raw_rgb.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return raw_rgb + gauss


def s_and_p(raw_rgb):
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(raw_rgb)
    # Salt mode
    num_salt = np.ceil(amount * raw_rgb.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in raw_rgb.shape]
    out[tuple(coords)] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * raw_rgb.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in raw_rgb.shape]
    out[tuple(coords)] = 0
    return out


def poisson(raw_rgb):
    vals = len(np.unique(raw_rgb))
    vals = 2 ** np.ceil(np.log2(vals))
    return np.random.poisson(raw_rgb * vals) / float(vals)


def speckle(raw_rgb):
    row, col, ch = raw_rgb.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    return raw_rgb + raw_rgb * gauss


def invert_colors(raw_rgb):
    return cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)


def laplacian(raw_rgb):
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Laplacian(img, cv2.CV_64F)


def double_saturation(raw_rgb):
    imghsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * 2
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
