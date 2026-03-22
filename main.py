from flask import Flask,render_template,request
import pickle
import os

app=Flask(__name__)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.pkl')

file=open(model_path,'rb')

regr=pickle.load(file)
file.close()

# Handle sklearn version compatibility issue
if not hasattr(regr, 'positive'):
    regr.positive = False
@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        myDict=request.form
        engine=float(myDict['Engine'])
        input_size = [engine]
        test_y_ = regr.predict([input_size])[0][0]
        #print(test_y_)
        return render_template('result.html',EMI=round(test_y_))
    return render_template('index.html')
    #return 'HEllo World'+str(test_y_)
    
if __name__=='__main__':
    print("Starting Flask app...")
    app.run(host='127.0.0.1', port=5000, debug=True)
