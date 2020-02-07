"""
Sample script that aligns and crops faces listed in text file. This is based on
5 landmarks listed alongside image name in text file.

TODO document expected format of landmarks read in as dictionary from pickle.

Created by Joseph Robinson on 24 January 2020.
"""
import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
import pandas as pd
import tqdm
from facebias.imutils import align_faces_affine


def align_faces(dir_in, dir_out, facial_points):
    """
    Reads in keys from landmarks dictionary as paths to face images relative to
    dir_in. Faces are then aligned with landmarks, and saved with same relative
    path but with respect to dir_out.
    :param dir_in:          root directory containing face data.
    :param dir_out:         root directory to write aligned faces to.
    :param facial_points:   landmarks to reference for alignment and cropping
    :return:
    """
    # make all subdirectories
    f_images = glob.glob(f"{dir_in}*males/n*/*.jpg")
    [
        Path(
            os.path.join(dir_out, str(Path(f_image.replace(dir_in, "")).parent))
        ).mkdir(exist_ok=True, parents=True)
        for f_image in f_images
    ]
    misses = []
    for f_image, points in tqdm.tqdm(facial_points.items()):
        f_image_in = dir_in + "/" + f_image
        f_image_out = dir_out + "/" + f_image
        if not len(points):
            shutil.copy(f_image_in, f_image_out)
            misses.append(f_image_in)
            continue
        elif Path(f_image_out).is_file():
            continue
        # elif len(points) > 1:
        #     print('multiple')
        #     continue
        image = cv2.imread(f_image_in)
        coords = points[0]["keypoints"]
        pts = (
            [coords["left_eye"]]
            + [coords["right_eye"]]
            + [coords["nose"]]
            + [coords["mouth_left"]]
            + [coords["mouth_right"]]
        )
        image = align_faces_affine(image, pts)

        cv2.imwrite(f_image_out, image)
    with open("misses.dat", "w") as f:
        for miss in misses:
            f.write(miss + "\n")


def read_landmarks(f_landmarks):
    """
    Reads file containing landmarks for image list. Specifically,
    <image name> x1 y1 ... xK yK is expected format.
    :param f_landmarks: text file with image list and corresponding landmarks
    :return:        landmarks as type dictionary.
    """
    landmarks_lut = {}
    with open(f_landmarks) as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        line = line.replace("\n", "").split("\t")
        landmarks_lut[line[0]] = [int(k) for k in line[1:]]

    return landmarks_lut


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align faces with 5 landmarks" "using affine transformation"
    )

    parser.add_argument(
        "-d",
        "--detections",
        default="../../data/bfw-data/bfw-fiducials-5pts.pkl",
        type=str,
        help="text file to read in and process",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="../../data/bfw-data/bfw/face-samples/",
        type=str,
        help="root directory of faces to align and crop",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="../../data/bfw-data/bfw/bfw-cropped-aligned/",
        type=str,
        help="output directory: save aligned and cropped",
    )
    args = parser.parse_args()

    # make out bin if does not exist
    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    landmarks = pd.read_pickle(args.detections)
    print("Read in {} landmarks from {}".format(len(landmarks), args.input))

    align_faces(args.input, args.outdir, landmarks)
