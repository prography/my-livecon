# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
    # 눈 세로 길이
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 눈 가로 길이
    C = dist.euclidean(eye[0], eye[3])

    # Eye Aspect Ratio 계산
    ear = (A + B) / (2.0 * C)
    return ear
 
# input: frame (type:image)
def detect_eyes(frame, detector, predictor) :
    # hyper parameters
    EYE_AR_THRESH = 0.27
    EYE_AR_CONSEC_FRAMES = 2

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (leb_start, leb_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (reb_start, reb_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

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
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        right_eyebrow = shape[reb_start:reb_end]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0

        # 여러 값에 따라 눈 종류 분리

        eye_shape = 'female '

        # 눈 감은 경우
        if ear < 0.2:
            eye_shape += 'closed'
        # 큰 눈의 경우
        elif ear >= 0.3:
            eye_shape += 'big'
        else:
            eye_shape += 'small'

        # 눈꼬리가 올라간 경우
        if eye_shape is not 'closed':
            if rightEye[0][1] < rightEye[3][1]:
                eye_shape += ' high'
            else:
                eye_shape += ' low'

        # 눈썹이 올라간 경우
        if right_eyebrow[0][1] < right_eyebrow[-1][1]:
            eyebrow_shape = 'high'
        else:
            eyebrow_shape = 'low'

    return eye_shape, eyebrow_shape