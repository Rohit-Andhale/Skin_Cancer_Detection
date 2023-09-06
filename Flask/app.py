from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np


app = Flask(__name__)

cell_type={ 4 : 'Cancer Cell Type is Melanocytic nevi',
    5: 'Cancer Cell  Type is Melanoma',
    2: 'Cancer Cell  Type is Benign keratosis-like lesions ',
    1: 'Cancer Cell  Type is Basal cell carcinoma',
    0: 'Cancer Cell  Type is Actinic keratoses',
    6: 'Cancer Cell  Type is Vascular lesions',
    3: 'Cancer Cell  Type is Dermatofibroma'}


model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = np.asarray(Image.open(img_path).resize((224, 224)))
	i = i/255
	i = i.reshape(1, 224, 224, 3)
	p = model.predict(i)
	p = np.argmax(p,axis=1)
	return cell_type[p[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "PLEASE AVOID SMOKING AND ALCOHOL CONSUMPTION..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)