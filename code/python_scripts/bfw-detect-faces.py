import glob
from tqdm import tqdm
import cv2
import pandas as pd
from mtcnn import MTCNN

dir_data = "../../data/bfw-data/samples/"
obin = "../../data/bfw-data/detected-face-parts-5pt.pkl"
imfiles = glob.glob(f"{dir_data}*males/n*/*.jpg")

print(f"Running face detector on {len(imfiles)} images")

detector = MTCNN()
faces_parts = {}
for img_path in tqdm(imfiles):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces_parts[img_path.replace(dir_data, "")] = detector.detect_faces(img)


pd.to_pickle(faces_parts, obin)
