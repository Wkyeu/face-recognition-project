import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

result = DeepFace.analyze(
    img_path="images/unknown_face.jpg",
    actions=['age', 'gender', 'emotion'],
    detector_backend="retinaface",
    enforce_detection=True
)

print(result)