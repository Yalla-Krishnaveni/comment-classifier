from flask import Flask, request, render_template
import pandas as pd
import pickle
df = pd.read_json("dataset.json", lines=True)


data = 'model.pkl'
model = pickle.load(open(data, 'rb'))
vect = pickle.load(open('transform.pkl', 'rb'))
tfidf = pickle.load(open('transform1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    comment = [x for x in request.form.values()]
    print(comment)

    x = df.content.values

    x = vect.transform(comment)
    x_tfidf = tfidf.transform(x)

    o = model.predict(x_tfidf)
    print(o)


    if o[0] == '0':
        return render_template('index.html', pred='Not a Bullying Comment')
    elif o[0] == '1':
        return render_template('index.html', pred='It is a Bullying Comment')
    else:
        return render_template('index.html', pred='Error Occurred in prediction')


if __name__ == '__main__':
    app.run(debug=True)
