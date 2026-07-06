import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

result = DeepFace.verify(
    img1_path="images/elon_1.jpg",
    img2_path="images/elon_2.jpg",
    model_name="VGG-Face",
    detector_backend="retinaface",
    enforce_detection=True
)

print(result)