from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2


def pil_image_to_base64(pil_image):
    _, buf = cv2.imencode(".png", pil_image)
    return base64.b64encode(buf)


def base64_to_pil_image(base64_img):
    buf_decode = base64.b64decode(base64_img)
    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)