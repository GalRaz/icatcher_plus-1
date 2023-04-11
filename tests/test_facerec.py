import pytest
import sys
from PIL import Image, ImageChops, ImageStat
from io import BytesIO
import face_recognition 

from test import FaceRec

def test_image_generation():
    """
    Test that an image can be generated given a bounding box and a frame/image file
    If the function works, the number of known faces will be exactly 1
    """
    fr = FaceRec()
    fr.get_ref_image("tests/test_images/test_image.png")
    
    test_img = face_recognition.load_image_file("tests/test_images/test_image.png")
    locs = fr.facerec_check(test_img)

    frame = face_recognition.load_image_file("tests/test_images/image_generation_test.png")
    
    test_bbox = locs

    fr = FaceRec()
    fr.generate_ref_image(test_bbox, frame)
    assert len(fr.known_faces) == 1


def test_get_ref_image():
    """
    Test that the image can be captured, will capture and log a "known face"
    If the function works, the number of known faces will be exactly 1.
    """
    fr = FaceRec()

    fr.get_ref_image("tests/test_images/test_image.png")
    assert len(fr.known_faces) == 1


def test_facerec_pass():
    """
    Test that the face recognition works with known faces, will register a match with a known face
    If the function works, it should return the location of a matching face
    """
    fr = FaceRec()

    fr.get_ref_image("tests/test_images/test_image.png")
    
    test_img = face_recognition.load_image_file("tests/test_images/test_image.png")
    locs = fr.facerec_check(test_img)

    assert locs != None


def test_facerec_fail():
    """
    Test that the face recognition works with known faces, will not register a match with an unknown face
    If the function works, the checker will return a None value if it doesn't find anything.
    """
    fr = FaceRec()

    fr.get_ref_image("tests/test_images/test_image.png")

    test_img = face_recognition.load_image_file("tests/test_images/test_image_facerec_fail.png")
    locs = fr.facerec_check(test_img)
    assert locs == None


