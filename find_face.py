import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

results = DeepFace.find(
    img_path="images/unknown_face.jpg",
    db_path="database",
    model_name="VGG-Face",
    detector_backend="retinaface",
    enforce_detection=True
)

print(results)