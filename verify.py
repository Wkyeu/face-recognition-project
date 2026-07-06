import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

result = DeepFace.verify(
    img1_path="images/elon_1.jpg",
    img2_path="images/elon_2.jpg",
    model_name="VGG-Face", #beginner model - could be Facenet, ArcFace, Dlib, SFace...
    detector_backend="retinaface",
    enforce_detection=True #if True, deepface will throw an error if it cant find a face in the image at all. If false it will proceed anyway.
)

print(result)