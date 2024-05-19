from fastapi import FastAPI
import tensorflow as tf

# Create an instance of FastAPI
app = FastAPI()

# Load a pre-trained TensorFlow model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# Define a route and its handler
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define another route for testing TensorFlow
@app.get("/predict/{image_path}")
def predict_image(image_path: str):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Expand the shape to (1, 224, 224, 3)
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=5)[0]
    return {"predictions": [(label, round(float(score), 4)) for (_, label, score) in decoded_predictions]}

# Run the FastAPI server using uvicorn
# Use the command: uvicorn filename:app --reload
# Replace 'filename' with the actual name of your Python script
# For example, if your script is named 'main.py', use: uvicorn main:app --reload
