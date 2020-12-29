import os, imutils
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from time import sleep
from distutils.version import LooseVersion
import skimage.color
import skimage.io
import skimage.transform
from tqdm import tqdm

orientations = ['vertical'
# , 'horizontal'
]
pipe_sizes   = [
    # 'one', 
    'three-fourth'
    # , 'one-half'
    ]
labels       = [
    # '25', '50', '75', 
    '100']
stdevstreams_dir = '../data/waterstreams'
preprocessed_dir = '../data/preprocessed-images'
thmasks_dir = '../data/generated-masks'

HEIGHT = WIDTH = 240

def main():
    dir_paths = []
    for orientation in orientations:
        for pipe_size in pipe_sizes:
            for label in labels:
                dir_path = os.path.join(orientation, pipe_size, label)
                dir_paths.append(dir_path)

    # pool = Pool(processes = 3)
    # pool.map(noisy_mask_on_frames, dir_paths)     

    noisy_mask_on_frames(dir_paths[0])     

def noisy_mask_on_frames(dir_path):
    mask_dir = os.path.join(thmasks_dir, dir_path)
    strm_dir  = os.path.join(preprocessed_dir, dir_path)
    rslt_dir = strm_dir.replace(preprocessed_dir, stdevstreams_dir)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rslt_dir, exist_ok=True)

    for frame_number in tqdm(range(3000)):
        image_filename = str(frame_number) + '.jpg'
        image_path = os.path.join(preprocessed_dir, dir_path, image_filename)
        output_path = os.path.join(thmasks_dir, dir_path, image_filename)

        # if (os.path.isfile(output_path)): continue

        image = cv.imread(image_path)
        image = to_gray(image)
        _, th = cv.threshold(image, 112, 255, cv.THRESH_BINARY)
        th = erode(th)
        th = erode(th)
        th = erode(th)
        th = dilate(th)

        th3c = np.stack((th,)*3, axis=-1)
        # resized = resize_image(th3c, max_dim=HEIGHT)[0]
        cv.imwrite(output_path, th3c[:,:,0])

        ### apply
        
        mask_path = os.path.join(mask_dir, image_filename)
        strm_path = os.path.join(strm_dir, image_filename)
        rslt_path = strm_path.replace(preprocessed_dir, stdevstreams_dir)

        # if (os.path.isfile(rslt_path)): continue

        mask = cv.imread(mask_path)
        strm = cv.imread(strm_path)

        # mask = resize_image(mask, max_dim=HEIGHT)[0]
        # strm = resize_image(strm, max_dim=HEIGHT)[0]

        extracted = apply_mask(strm, mask[:,:,0])
        extracted = resize_image(extracted, max_dim=HEIGHT)[0]

        cv.imwrite(rslt_path, extracted[:,:,0])
        cv.imwrite(mask_path, mask[:,:,0])

def to_one_channel(image):
    image = image[:, :, 0]
    height, width = image.shape
    image = image.reshape((height, width, 1))
    return image

def std_stream(stdev, save_to_path):
    resized    = resize_image(stdev, min_dim=HEIGHT, max_dim=HEIGHT)[0]
    save_to_path = save_to_path.replace(thmasks_dir, stdevstreams_dir)
    cv.imwrite(save_to_path, resized)

def resize_stream(to_resize, save_to_path):
    resized = resize_image(to_resize, min_dim=HEIGHT, max_dim=HEIGHT)[0]
    save_to_path = save_to_path.replace(preprocessed_dir, stdevstreams_dir)
    cv.imwrite(save_to_path, resized)

def noisy_mask_on_image(dir_path):
    sleep(.1)
    input_dir  = os.path.join(preprocessed_dir, dir_path)
    output_dir = os.path.join(thmasks_dir  , dir_path)

    image_filenames = filenames_in(input_dir)

    for image_filename in image_filenames:
        image_path = os.path.join(input_dir, image_filename)
        image      = cv.imread(image_path)
        
        gray     = to_gray(image)
        gray_inv = invert_color(gray)
        black_and_white = adaptive_threshold(gray_inv)

        horizontal_lines, vertical_lines = get_lines(black_and_white)
        lines      = horizontal_lines + vertical_lines
        lines_inv  = invert_color(lines)
        
        no_squares     = remove_squares(lines_inv)
        no_squares_inv = invert_color(no_squares)

        no_lines   = black_and_white - lines
        no_bg      = no_lines - no_squares_inv

        opened     = opening(no_bg)
        nthreshold = neighborhood_threshold(opened)
        filled     = fill_large_contour(nthreshold)
        noisy_mask = filled

        output_path = os.path.join(output_dir, image_filename)
        cv.imwrite(output_path, noisy_mask)    
        print("processed: " + output_path)
        
def filenames_in(dir_path):
    (_, _, filenames) = next(os.walk(dir_path))
    return filenames

def to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def get_lines(image, scale = 5):
    horizontal = np.copy(image)
    vertical   = np.copy(image)

    rows    = vertical.shape[0]
    columns = horizontal.shape[1]

    horizontal_size = columns // scale
    vertical_size   = rows    // scale

    horizontal_struct_elem = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    vertical_struct_elem   = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))

    ### opening

    horizontal = cv.erode(horizontal, horizontal_struct_elem + 3)
    horizontal = cv.dilate(horizontal, horizontal_struct_elem)
    
    vertical   = cv.erode(vertical  , vertical_struct_elem  + 3)
    vertical   = cv.dilate(vertical  , vertical_struct_elem)

    return horizontal, vertical

def opening(image, kernel = (3, 3)):
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def closing(image, kernel = (3, 3)):
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

def neighborhood_threshold(image, kernel = 9, threshold_percent = 0.05, step = 4):
    rows    = image.shape[0]
    columns = image.shape[1]
        
    rows    = rows    // kernel
    columns = columns // kernel

    white     = 255
    threshold = white * threshold_percent

    for row in range(rows):
        for column in range(columns):
            x1 = row * kernel
            y1 = column * kernel
            x2 = x1 + kernel
            y2 = y1 + kernel

            neighborhood = image[x1:x2, y1:y2]

            if (np.mean(neighborhood) > threshold):
                image[x1:x2, y1:y2] = 1
            else:
                image[x1:x2, y1:y2] = 0

    return image

def fill_large_contours(image, count = 3):
    _, contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)    
    max_contour  = max(contours, key=cv.contourArea)
    # cntrsSorted = sorted(contours, key=lambda x: cv.contourArea(x))[count]
    
    cv.drawContours(image, contours     , -1,   0, thickness=cv.FILLED)
    cv.drawContours(image, [max_contour], -1, 255, thickness=cv.FILLED)
    
    return image

def invert_color(image):
    return cv.bitwise_not(image)

def adaptive_threshold(image, blocksize = 19, c = 2):
    return cv.adaptiveThreshold (
                                  image
                                , 255
                                , cv.ADAPTIVE_THRESH_MEAN_C
                                , cv.THRESH_BINARY
                                , blocksize
                                , c
                                )

def remove_squares(image):
    edged = cv.Canny(image, 85, 255)
    
    contours = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    squares = np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    for contour in contours:
        if is_squarish(contour):
            cv.drawContours(squares, [contour], -1, 0, -1)
    
    no_squares = cv.bitwise_and(image, image, mask=squares)

    return no_squares

def sharpen(image):
    kernel    = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def unsharpen(image):
    blurred     = cv.GaussianBlur(image, (3,3), 10.0)
    unsharpened = cv.addWeighted(image, 1.5, blurred, -0.5, 0, image)
    return unsharpened

def is_squarish(contour):
    peri   = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    return len(approx) > 4 and len(approx < 8) 

def erode(image, val_type = 0, erosion_size = 3):
    if val_type == 0:
        erosion_type = cv.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv.MORPH_ELLIPSE
    
    element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    
    return cv.erode(image, element)

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def dilate(image, val_type = 0, erosion_size = 3):
    if val_type == 0:
        erosion_type = cv.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv.MORPH_ELLIPSE
    
    element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    
    return cv.dilate(image, element)

def apply_mask(image, mask):
    return cv.bitwise_and(image, image, mask=mask)

if __name__ == '__main__':
    main()