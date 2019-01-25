from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2

def mouth_main(frame, detector, predictor):
    (lip_start, lip_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        lip = shape[lip_start: lip_end]

        if abs(lip[14][1] - lip[18][1]) > 10:
            lip_categ = 'open'
        else:
            # 입을 다물고 있는 경우 입꼬리가 올라갔는지 내려갔는지 판별
            base = (float)(lip[14][1] + lip[18][1])/2
            end = lip[0][1]
            if float(end) > base:
                lip_categ = 'closed low'
            elif float(end) < base:
                lip_categ = 'closed high'
            else:
                lip_categ = 'closed line'
    return lip_categ