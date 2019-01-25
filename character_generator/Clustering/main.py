# import the necessary packages
from imutils.face_utils import FaceAligner
import argparse
import imutils, dlib, cv2
from PIL import Image

from eyeglasses_detector.detect import detect_eyeglasses
from eyes_classifier.detect import detect_eyes
from face_classifier.detect import detect_skincolor
from mouth_classifier.detect import mouth_main

import time

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='assets/images/glasses2.jpg', help='path to input image')
    parser.add_argument('--output_path', type=str, default='outputs', help='path to save result character images')
    parser.add_argument('--shape_predictor', type=str, default='assets/classifiers/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--face_clf_path', type=str,
                        default='assets/classifiers/face_classifier.xml', help='classifier path to localize frontal face')
    config = parser.parse_args()
    return config

def align_face(image, detector, predictor):
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        faceAligned = fa.align(image, gray, rect)

    return faceAligned

def extract():
    config = get_config()

    # get image
    image = cv2.imread(config.image_path)

    # call face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.shape_predictor)
    image = align_face(image, detector, predictor)

    # define face cascade classifier
    face_cascade = cv2.CascadeClassifier(config.face_clf_path)

    # convert color image to gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        face_color = image
    else:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            face_color = image[y:y+h, x:x+w]

    # detect eye glasses
    # pass image path as function parameter
    glasses = detect_eyeglasses(config.image_path)

    # detect eyes shapes and eyebrow shape
    eye_shape, eyebrow_shape = detect_eyes(face_color, detector, predictor)

    # detect skin color category
    skincolor_categ = detect_skincolor(face_color)

    # detect mouth shape --> open / upper / lower
    mouth_shape = mouth_main(face_color, detector, predictor)

    features = [glasses.item(), eye_shape, eyebrow_shape, skincolor_categ, mouth_shape]
    return features

def match_image_properties(image):
    result = image.convert('RGBA').resize((480, 500))
    return result

# merge
def gen_character(features):
    [glasses, eye_shape, eyebrow_shape, skin_color_categ, mouth_shape] = features

    # choose eyeglasses
    if glasses == 1:
        eyeglasses = match_image_properties(Image.open('dataset/glasses/glasses.png'))

    # choose eyes in Eyes dataset
    eye = match_image_properties(Image.open('dataset/eyes/%s.png' % eye_shape))

    # choose eyebrow in Eyebrow dataset
    eyebrow = match_image_properties(Image.open('dataset/eyebrow/%s eyebrow.png' % eyebrow_shape))

    # choose face based on skin color in face dataset
    face = match_image_properties(Image.open('dataset/face/%s.png' % skin_color_categ))

    # choose mouth shape on mouth dataset
    mouth = match_image_properties(Image.open('dataset/mouth/%s.png' % mouth_shape))

    blended1 = Image.alpha_composite(face, eye)
    blended2 = Image.alpha_composite(blended1, eyebrow)
    blended3 = Image.alpha_composite(blended2, mouth)
    blended3.save('merged.png')
    print("[*]Generate character COMPLETED!")

if __name__ == '__main__':
    start_time = time.time()
    features = extract()
    print("Extraced features:", features)
    gen_character(features)
    end_time = time.time()
    print("Inference time:%.3f" % (end_time-start_time))
