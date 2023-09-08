from flask import Flask, request, render_template
from credit_card_fraud_detection.pipeline.stage_3_predict import Predict
from  credit_card_fraud_detection.constants.job import job
from credit_card_fraud_detection.constants.category import category
from credit_card_fraud_detection.constants.city import city
from credit_card_fraud_detection.constants.merchant import merchant
from credit_card_fraud_detection.constants.state import state


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret_base_key"



@app.route('/')
def home():
    return render_template('index.html', job=job, category=category, city=city, merchant= merchant, state=state)

@app.route('/', methods=['GET','POST'])
def validator():
    if request.method=='POST':
        dictionary={}
        dictionary["cc_num"]=int(request.form["cc_num"])
        dictionary["merchant"]=(request.form["merchant"])
        dictionary["category"]=(request.form["category"])
        dictionary["amt"]=float(request.form["amt"])
        dictionary["gender"]=(request.form["gender"])
        dictionary["city"]=(request.form["city"])
        dictionary["state"]=(request.form["state"])
        dictionary["zip"]=int(request.form["zip"])
        dictionary["lat"]=float(request.form["lat"])
        dictionary["long"]=float(request.form["long"])
        dictionary["city_pop"]=int(request.form["city_pop"])
        dictionary["job"]=(request.form["job"])
        dictionary["dob"]=(request.form["dob"])
        dictionary["unix_time"]=int(request.form["unix_time"])
        dictionary["merch_lat"]=float(request.form["merch_lat"])
        dictionary["merch_long"]=float(request.form["merch_long"])
        print(dictionary)
        obj = Predict()
        result=obj.main(dictionary=dictionary)
        print(result)
        return render_template('index.html', result=result, job=job, category=category, city=city, merchant= merchant, state=state)

    return render_template('index.html',job=job, category=category, city=city, merchant= merchant, state=state)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=False)