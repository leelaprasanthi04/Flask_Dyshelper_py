# #********** receiving local audio or recorded audio along with the image
# import os
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Directory to save uploaded files
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Allowed file extensions
# ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "aac"}
# ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# def allowed_file(filename, allowed_extensions):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "audio" not in request.files or "image" not in request.files:
#         return jsonify({"error": "Both audio and image files are required"}), 400

#     audio = request.files["audio"]
#     image = request.files["image"]

#     # Save audio file
#     if audio and allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
#         audio_filename = secure_filename(audio.filename)
#         audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
#         audio.save(audio_path)
#     else:
#         return jsonify({"error": "Invalid audio file format"}), 400

#     # Save image file
#     if image and allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
#         image_filename = secure_filename(image.filename)
#         image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
#         image.save(image_path)
#     else:
#         return jsonify({"error": "Invalid image file format"}), 400

#     return jsonify({
#         "message": "Files uploaded successfully",
#         "audio_url": f"/uploads/{audio_filename}",
#         "image_url": f"/uploads/{image_filename}"
#     }), 200

# if __name__ == "__main__":
#     app.run(debug=True)


#******** working but iamge is not opening in the browser

# import os
# from flask import Flask, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from pillow_heif import open_heif
# from PIL import Image
# app = Flask(__name__)

# # Directory to save uploaded files
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def convert_heic_to_jpeg(heic_path, jpeg_path):
#     heif_image = open_heif(heic_path)
#     img = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
#     img.save(jpeg_path, format="JPEG")

# # Allowed file extensions
# ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "aac"}
# ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "heic"}

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# def allowed_file(filename, allowed_extensions):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "audio" not in request.files or "image" not in request.files:
#         return jsonify({"error": "Both audio and image files are required"}), 400

#     audio = request.files["audio"]
#     image = request.files["image"]

#     if not allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
#         return jsonify({"error": "Invalid audio file format"}), 400
#     if not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
#         return jsonify({"error": "Invalid image file format"}), 400

#     # Save audio file
#     audio_filename = secure_filename(audio.filename)
#     audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
#     audio.save(audio_path)

#     # Save image file
#     image_filename = secure_filename(image.filename)
#     image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
#     image.save(image_path)

#     # Get server URL dynamically
#     server_url = request.host_url.rstrip('/')
#     result = {"message": "Files uploaded successfully",
#         "audio_url": f"{server_url}/uploads/{audio_filename}",
#         "image_url": f"{server_url}/uploads/{image_filename}"}
#     print(result)
#     return jsonify({
#         "message": "Files uploaded successfully",
#         "audio_url": f"{server_url}/uploads/{audio_filename}",
#         "image_url": f"{server_url}/uploads/{image_filename}"
#     }), 200

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



#####********* working code, audio and image both are coming 

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pillow_heif import open_heif
from PIL import Image
import uuid
import speechtotext
import process1 
app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def convert_heic_to_jpeg(heic_path, jpeg_path):
#     """ Convert HEIC to JPEG """
#     heif_image = open_heif(heic_path)
#     img = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
#     img.save(jpeg_path, format="JPEG")
#     os.remove(heic_path)  # Remove original HEIC file

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "aac"}
#ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "heic"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

# def display_image(image_path):
#     """ Display the image using OpenCV (or save a temporary preview) """
#     try:
#         img = cv2.imread(image_path)
#         if img is not None:
#             cv2.imshow("Uploaded Image", img)
#             cv2.waitKey(3000)  # Show for 3 seconds
#             cv2.destroyAllWindows()
#         else:
#             print(f"Could not open image: {image_path}")
#     except Exception as e:
#         print(f"Error displaying image: {str(e)}")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files or "image" not in request.files:
        return jsonify({"error": "Both audio and image files are required"}), 400

    audio = request.files["audio"]
    image = request.files["image"]
    
    filename = str(uuid.uuid4()) + os.path.splitext(image.filename)[1]
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    # Save the uploaded image
    image.save(image_path)
    print (f"Image saved successfully at {image_path}", 200)

    if not allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({"error": "Invalid audio file format"}), 400
    # if not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
    #     return jsonify({"error": "Invalid image file format"}), 400

    # Save audio file
    audio_filename = secure_filename(audio.filename)
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
    audio.save(audio_path)

    # Save image file
    # image_filename = secure_filename(image.filename)
    # image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
    # image.save(image_path)

    print(f"Uploaded Image: {image_path}")  # Print the filename

    # Convert HEIC to JPEG if needed
    # if image.filename.lower().endswith(".heic"):
    #     jpg_filename = image_filename.rsplit(".", 1)[0] + ".jpg"
    #     jpg_path = os.path.join(app.config["UPLOAD_FOLDER"], jpg_filename)
    #     convert_heic_to_jpeg(image_path, jpg_path)  # Convert and delete HEIC
    #     image_filename = jpg_filename  # Update filename to return in response
    #     image_path = jpg_path  # Update path for display

    # Display the image
    # display_image(image_path)
    # app_audioandfile.py, process1.py, speechtotext.py, models, ocr.json, 
    extracted_text = process1.detect_text1(image_path)
    cleaned_text = extracted_text.replace("\n", " ")
    feature_array = process1.get_feature_array(cleaned_text)
    prediction = process1.dys_pred(feature_array)

    audio_to_text_input = speechtotext.process_file(audio_path)
    feedback = speechtotext.compare_texts(audio_to_text_input, cleaned_text)
    cleaned_feedback_text = feedback.replace("\n", " ")

    # Get server URL dynamically
    # server_url = request.host_url.rstrip('/')

    result = {
            "Extracted_Text": cleaned_text,
            "Text from the Audio": audio_to_text_input,
            "Feature_Array": feature_array,
            "Disease_Presence": prediction,
            "FeedBack" : cleaned_feedback_text
        }

    print(result)

    return jsonify({
            "Text from the Image": cleaned_text,
            "Text from the Audio": audio_to_text_input,
            "Feature_Array": feature_array,
            "Disease_Presence": prediction,
            "FeedBack" : feedback
        })
    # result = {
    #     "message": "Files uploaded successfully",
    #     "audio_url": f"{server_url}/uploads/{audio_filename}",
    #     "image_url": f"{server_url}/uploads/{image_path}"
    # }
    # print(result)
    # return jsonify(result), 200

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
