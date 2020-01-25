from __future__ import print_function

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.linalg import inv, lstsq, matrix_rank as rank, norm
from skimage import transform

from .iotools import exists


def get_image_size_from_file(f_image):
    """
    :returns size of image stored at f_image
    """
    im = Image.open(f_image)
    return im.size  # (width,height) tuple


def bgr2rgb(im_bgr):
    """
    Wrapper for cv2 color conversion color BGR (cv2 format) to color RGB.
    """
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


def gray2jet(img):
    """[0,1] grayscale to [0.255] RGB"""
    jet = plt.get_cmap("jet")
    return np.uint8(255.0 * jet(img)[:, :, 0:3])


def write(img_file, arr):
    """Read image from file
    :param arr: image array to save to disk.
    :type arr: np.array
    :param img_file: filepath of image to open and return
    :type img_file: string
    """
    return mpimg.imsave(img_file, arr=arr)


def resize(image, height, width):
    return transform.resize(image, (height, width))


def read(img_file, as_rgb=False):
    """Read image from file
    :param as_rgb:
    :param img_file: filepath of image to open and return
    :type img_file: string
    """
    if not exists(img_file):
        return None
    if as_rgb:
        gray2rgb(mpimg.imread(img_file))
    else:
        return mpimg.imread(img_file)


def is_rgb(image):
    """
    Return True if image is RGB (ie 3 channels) for pixels in WxH grid
    """
    return len(image.shape) == 3 and image.shape[-1] == 3


def gray2rgb(image):
    if is_rgb(image):
        return image
    return np.stack((image, image.shape[0], image.shape[1]), axis=2)


def img2gray(image):
    """ Convert image to grayscale """
    if is_rgb(image):
        return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)
    else:
        return image


def resize_and_flatten(image, height=30, width=30):
    """ resize image and flatten (vectorize) """
    row_res = cv2.resize(image, (height, width),
                         interpolation=cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image, (height, width),
                         interpolation=cv2.INTER_AREA).flatten("F")
    return row_res, col_res


class MatlabHandleException(Exception):
    def __str__(self):
        return "In File {}:{}".format(__file__, super.__str__(self))


def trans_fwd(trans, uv):
    """ Apply affine transform 'trans' to uv

    :param trans: 3x3 np.array, transform matrix
    :param uv: Kx2 np.array, each row is a pair of coordinates (x, y)
    :return: Kx2 np.array, each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def trans_inv(trans, uv):
    """ Apply the inverse of affine transform 'trans' to uv

    :param trans: 3x3 np.array, transform matrix
    :param uv: Kx2 np.array, each row is a pair of coordinates (x, y)
    :return: Kx2 np.array, row is inverse-transformed coordinates (x, y)
    """
    t_inv = inv(trans)
    xy = trans_fwd(t_inv, uv)
    return xy


def find_non_reflective_similarity(uv, xy):
    """ Find Non-reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    :param uv: Kx2 np.array, source points each row is a coordinate pair (x, y)
    :param xy: Kx2 np.array, each row is a pair of inverse-transformed
    :returns:   trans: transform matrix from uv to xy, 3x3
                type: np.array
                trans_inv: inverse of trans, transform matrix from xy to uv, 3x3
                type: np.array

    """
    options = {"K": 2}

    n_landmarks = options["K"]
    n_pts = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((n_pts, 1)), np.zeros((n_pts, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((n_pts, 1)), np.ones((n_pts, 1))))
    x_tensor = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    u_matrix = np.vstack((u, v))

    # We know that X * r = U (u_matrix)
    if rank(x_tensor) >= 2 * n_landmarks:
        r, _, _, _ = lstsq(x_tensor, u_matrix)
        r = np.squeeze(r)
    else:
        raise Exception("points2transform:twoUniquePointsReq")

    sc, ss, tx, ty = r[0], r[1], r[2], r[3]

    t_inv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    inverse_tensor = inv(t_inv)
    inverse_tensor[:, 2] = np.array([0, 0, 1])

    return inverse_tensor, t_inv


def find_similarity(uv, xy):
    """ Find Reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans

        Matlab:
        ----------
        The similarities are a superset of the non-reflective similarities, as
        it may also include reflection.

     let sc = s*cos(theta)
     let ss = s*sin(theta)

                       [ sc -ss
     [u v] = [x y 1] *   ss  sc
                         tx  ty]

              OR

                       [ sc  ss
     [u v] = [x y 1] *   ss -sc
                         tx  ty]

    Algorithm:
     1) Solve for trans1, a non-reflective similarity.
     2) Reflect the xy data across the Y-axis,
        and solve for trans2r, also a non-reflective similarity.
     3) Transform trans2r to trans2, undoing the reflection done in step 2.
     4) Use TFORMFWD to transform uv using both trans1 and trans2,
        and compare the results, Returns the transformation corresponding
        to the smaller L2 norm.

     Need to reset options.K to prepare for calls to findNonreflectiveSimilarity
     This is safe because we already checked that there are enough point pairs.

    :param uv: Kx2 np.array, source points each row is a coordinate pair (x, y)
    :param xy: Kx2 np.array, each row is a pair of inverse-transformed
    :returns:   trans: transform matrix from uv to xy, 3x3
                type: np.array
                trans_inv: inverse of trans, transform matrix from xy to uv, 3x3
                type: np.array
    """
    # Solve for trans1
    trans1, trans1_inv = find_non_reflective_similarity(uv, xy)

    # manually reflect the xy data across the Y-axis
    xy_r = xy
    xy_r[:, 0] = -1 * xy_r[:, 0]

    # Solve for trans2
    trans2r, trans2r_inv = find_non_reflective_similarity(uv, xy_r)

    # manually reflect the tform to undo the reflection done on xyR
    trans_reflect_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, trans_reflect_y)

    # Figure out if trans1 or trans2 is better
    xy1 = trans_fwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = trans_fwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """ Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    :param src_pts: source points [Kx2], each row is a coordinate pair (x, y)
    :type src_pts: np.array
    :param dst_pts: destination points [Kx2], transformed coordinate pair (x, y)
    :type dst_pts: np.array
    :param reflective: use reflective similarity transform if True; else, use
    non-reflective similarity transform
    :returns:   trans: transform matrix from uv to xy, 3x3
                type: np.array
                trans_inv: inverse of trans, transform matrix from xy to uv, 3x3
                type: np.array
    """
    if reflective:
        trans, trans_inverse = find_similarity(src_pts, dst_pts)
    else:
        trans, trans_inverse = find_non_reflective_similarity(src_pts, dst_pts)

    return trans, trans_inverse


def cvt_tform_mat_for_cv2(trans):
    """
    Convert Transform Matrix 'trans' into 'cv2_trans' which could be directly
    used by cv2.warpAffine():

            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    :param trans: 3x3 np.array, transform matrix from uv to xy
    :returns: cv2_trans: transform matrix from src_pts to dst_pts, could be
    used for cv2.warpAffine(), 2x3 np.array
    """
    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """ Find Similarity Transform Matrix 'cv2_trans' which could be directly
    used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    :param src_pts: source points [Kx2], each row is a coordinate pair (x, y)
    :type src_pts: np.array
    :param dst_pts: destination points [Kx2], transformed coordinate pair (x, y)
    :type dst_pts: np.array
    :param reflective: use reflective similarity transform if True; else, use
    non-reflective similarity transform
    :return: cv2_trans: 2x3 np.array, transform matrix from src_pts to dst_pts,
    could be directly used for cv2.warpAffine()
    """
    trans, _ = get_similarity_transform(src_pts, dst_pts,
                                        reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans


def align_faces_affine(
        src_img,
        src_pts,
        crop_size=(108, 124),
        ref_pts=(
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
        ),
):
    """
    Aligns via affine transformation.
    :param ref_pts:
    :param crop_size:
    :param src_img: Facial image to align
    :param src_pts: source points [Kx2], each row is a coordinate pair (x, y)
    :type src_pts: np.array
    :return: aligned face.
    """

    src_pts = np.array(src_pts).reshape(-1, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img.astype(np.uint8), tfm, crop_size)

    return face_img
