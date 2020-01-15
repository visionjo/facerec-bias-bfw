import glob
from tqdm import tqdm
import cv2
import pandas as pd
from mtcnn import MTCNN

dir_data = '../../data/bfw-data/samples/'
imfiles = glob.glob(f"{dir_data}*males/n*/*.jpg")

print(f"Running face detector on {len(imfiles)} images")
# traingen= image_batch_generator(imfiles, batchsize=128)
# detector = face_recognition.api.cnn_face_detector()


# for _, img_path in tqdm(t):
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     # img = face_recognition.load_image_file(img_path)
#     # img_pil = Image.fromarray(img).convert("RGB")
#     # img_pil = img_pil.resize(img_size)
#     # img = cv2.resize(img,(img.shape[0]*2, img.shape[1]*2))
#     # for img, impath in zip(imgs, img_paths):
#
#     faces_parts3.append((detector.detect_faces(img), img_path))
#     # faces_parts3.append((face_recognition.face_landmarks(img), img_path))
detector = MTCNN()
faces_parts = {}
for img_path in tqdm(imfiles):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # img = face_recognition.load_image_file(img_path)
    # img_pil = Image.fromarray(img).convert("RGB")
    # img_pil = img_pil.resize(img_size)
    # for img, impath in zip(imgs, img_paths):
    
    faces_parts[img_path.replace(dir_data, '')] = detector.detect_faces(img)
    # append((face_recognition.face_landmarks(img), img_path))


pd.to_pickle(faces_parts, 'face-parts.pkl')
# df = pd.DataFrame().from_dict(faces_parts)
# pd.to(faces_parts, 'face-parts.pkl')
# for batch in image_batch_generator(imfiles):
#     pass


# t=[f for f in faces_parts if not f[0]]

