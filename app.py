import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---- Config ----
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "cat_dog_model.h5"
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.secret_key = "replace-with-a-secret-key"  # needed for flash messages
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Load model once on startup ----
model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(img_path: str) -> np.ndarray:
    """Load and preprocess image to shape (1, 224, 224, 3) scaled 0–1."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_label(img_path: str):
    """Return ('Cat' or 'Dog', probability_for_dog_float_0_1)."""
    batch = preprocess(img_path)
    prob = float(model.predict(batch, verbose=0)[0][0])  # sigmoid output
    label = "Dog" if prob >= 0.5 else "Cat"
    return label, prob

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    image_url = None
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload JPG/PNG/WEBP.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        # Avoid overwriting by appending a number if necessary
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(save_path):
            filename = f"{base}_{counter}{ext}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            counter += 1

        file.save(save_path)

        label, prob = predict_label(save_path)
        prediction = label
        probability = prob
        image_url = url_for("uploaded_file", filename=filename)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        image_url=image_url
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return app.send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # For local dev. In production, use gunicorn/uwsgi, etc.
    app.run(host="0.0.0.0", port=5000, debug=True)

