from pathlib import Path

import fiftyone as fo
import pandas as pd
from fiftyone import Sample
from fiftyone.core.labels import Classification, Detection, Keypoint
from skimage import io

dataset = fo.Dataset("bfw")
kpts_ref = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]


def normalize_detection(fpath, det):
    """
    fiftyone accepts BB coordinates and keypoints as normalized in pixel space
    locations (i.e., [0, 1]).

    :param fpath:
    :type fpath:
    :param det:
    :type det:
    :return:
    :rtype:
    """
    im = io.imread(fpath)
    h, w = float(im.shape[0]), float(im.shape[1])
    kpts = det["keypoints"]

    kpts_tuples = [(kpts[r][0] / w, kpts[r][1] / h) for r in kpts_ref]
    kpts_dic = {r: kpt for r, kpt in zip(kpts_ref, kpts_tuples)}
    box = det["box"]
    x1, y1, x2, y2 = box
    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

    confidence = det["confidence"]

    return rel_box, kpts_dic, confidence


dir_data = str(Path.home() / "bfw/uncropped-face-samples/")
file_detections = str(
    Path.home() / "/WORK/src/facebias/data/bfw/detected-face-parts-5pt.pkl"
)

detections = pd.read_pickle(file_detections)
path_subgroups = Path(dir_data).glob("*_*males")


for p_subgroup in path_subgroups:
    print(p_subgroup)
    subgroup = p_subgroup.name.split("_")[0][0] + p_subgroup.name.split("_")[1][0]
    gender = subgroup[1]
    ethnicity = subgroup[0]

    path_subjects = p_subgroup.glob("n??????")

    for p_subject in path_subjects:
        print(p_subject)
        subject_id = p_subject.name

        path_faces = p_subject.glob("*.jpg")
        for p_face in path_faces:
            print(p_face)
            face_id = p_face.name
            ref = "/".join(str(p_face).split("/")[-3:])

            detection = detections[ref] if ref in detections else []

            if len(detection):
                bb, kpts_dic, confidence = normalize_detection(
                    str(p_face), detection[0]
                )
            else:
                bb, kpts_dic, confidence = None, None, None
            # p_face_ = str(p_face).replace("bfw-cropped-aligned/", "face-samples/")
            sample = Sample(filepath=str(p_face))
            sample["identity"] = Classification(label=subject_id)
            sample["gender"] = Classification(label=gender)
            sample["ethnicity"] = Classification(label=ethnicity)
            sample["prediction"] = Detection(
                label=subject_id, bounding_box=bb, confidence=confidence
            )
            # sample['keypoints'] = Keypoint
            if kpts_dic is not None:
                for k, v in kpts_dic.items():
                    keypoint = Keypoint(label=k, points=[v])
                    sample[k] = keypoint
                    del keypoint

            dataset.add_sample(sample)

dataset.save()
session = fo.launch_app(dataset)
session.wait()
