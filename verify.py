from deepface import DeepFace

result = DeepFace.verify(
    img1_path="images/elon_1.jpg",
    img2_path="images/elon_2.jpg",
    model_name="VGG-Face",
    detector_backend="opencv",
    enforce_detection=True
)

print(result)