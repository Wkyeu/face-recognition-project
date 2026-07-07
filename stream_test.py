import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

DeepFace.stream(
    db_path="database",
    model_name="VGG-Face",
    detector_backend="opencv",
    time_threshold=5,
    frame_threshold=5
)