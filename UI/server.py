from flask import Flask, request, render_template, url_for
import pandas as pd
import pickle
import numpy as np
with open('model.pkl','rb') as f:
    classifier = pickle.load(f)
with open('scaling.pkl','rb') as s:
    sc = pickle.load(s)

dataset=pd.read_csv("Clean_MSD.csv")
dataset_artist=pd.read_csv("popular_artist.csv")

app = Flask(__name__)

@app.route("/main")
def home():
    return render_template("index.html")

@app.route("/song",methods=["POST","GET"])
def song():
    if request.method == "POST":
        data_form = request.form.get("Year")
        df=dataset[dataset['year'] == float(data_form)]
        return render_template("song.html",dataset=df, out=True)
    else:
        return render_template("song.html")
        

@app.route("/prediction",methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        dataform = request.form
        l=list()
        for i in dataform:
            if i == "song":
                pass
            elif i == "artist":
                l.append(dataform[i])
            else:
                l.append(float(dataform[i]))
        print(l)
        #l=[1,1,1,1,1,1,1,1,1,1,1,1,1]
        l=np.array(l)
        l=l.reshape(1,-1)
        l = sc.transform(l)
        pred=classifier.predict(l)
        return render_template("prediction.html",s = pred[0],out=True)
    else:
        return render_template("prediction.html")

@app.route("/visual")
def visual():
    return render_template("visual.html",dataset=dataset_artist)


if __name__ == "__main__":
    app.run(debug=True)
