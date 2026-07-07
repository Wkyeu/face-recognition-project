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

#to grab the top match 
df = results[0]
if len(df) > 0:
    best_match = df.iloc[0]
    print(f"Best match: {best_match['identity']} (distance: {best_match['distance']:.3f})")
else: 
    print("No match found in database")