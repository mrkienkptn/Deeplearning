import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, send_from_directory, send_file
# from flask_resful import reqparse
import os
from werkzeug.utils import secure_filename
from process_normal_img import make_full_image
from process_face_img import generate
from PIL import Image


app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "./static/images/upload"
app.config["IMAGE_GEN"] = "./static/images/gen"
app.config["ALLOW_IMAGE_EXTENTIONS"] = ["png", "jpg", "jpeg"]
app.config["MAX_SIZE"] = 2000*2000
app.config["MIN_SIZE"] = 10

def allow_image(filename):
	if not "." in filename:
		return False

	ext = filename.rsplit(".", 1)[1]
	if ext.lower() in app.config["ALLOW_IMAGE_EXTENTIONS"]:
		return True

	else:
		return False


def allow_image_size(size):

	if int(size) <= app.config["MAX_SIZE"] and int(size) >= app.config["MIN_SIZE"]:
		return True

	else:
		return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

	if request.method == "POST":

		if request.files:

			image = request.files["image"]

			if image.filename == "":
				return redirect(request.url)
			print(request.cookies.get("filesize"))

			if not allow_image_size(request.cookies.get("filesize")):
				
				return redirect(request.url , data = "Image size is exceeded")

			if allow_image(image.filename):

				secure_name = secure_filename(image.filename)

				image.save(os.path.join(app.config["IMAGE_UPLOADS"], secure_name))


			else:
				return render_template("index.html", data="This extension is not allow")

			#process image
				

			return redirect(request.url)
	data = {
			'image_name' :request.cookies.get("filename"),
			'status' : 'Upload Successfully'
		}

	return render_template("index.html", data=data)

@app.route("/process-image/<path:path>", methods=["POST"])
def process_image(path):
	name = secure_filename(path)
	type_im = request.form.getlist('picture_type')[0]
	print("request: ",type_im)

	image_path = os.path.join(app.config["IMAGE_UPLOADS"], name)
	
	if type_im=='normal_im':

		gen_image = make_full_image(image_path)

		gen_image.save(os.path.join(app.config["IMAGE_GEN"], name))

	if type_im == 'face':
		gen_image = generate(image_path)

		gen_image.save(os.path.join(app.config["IMAGE_GEN"], name))

	data = {
			'image_name_down' :request.cookies.get("filename"),
			'status' : 'Process Completed'

		}
	return render_template('index.html', data=data)

@app.route("/download-image/<path:path>")
def download_image(path):
	name = secure_filename(path)
	if not os.path.exists(os.path.join(app.config["IMAGE_GEN"], name)):
		data = {
			'image_name': '',
			'status':'No thing to download'
		}
		return render_template('index.html', data=data)
	# return send_from_directory(app.config["IMAGE_GEN"], name, as_attachment=True)
	return send_file(os.path.join(app.config["IMAGE_GEN"], name), as_attachment=True)

@app.route("/")
def index():
	data = {
			'image_name': '',
			'status':''
		}
	return render_template("index.html", data=data)

