from flask import Flask, jsonify
import test_model as tm

app = Flask(__name__)

@app.route('/attractions/<string:userId>/', methods=['GET'])
def user_view(userId):
    result = tm.getPredictions(10, userId)        
    return jsonify(result)

if __name__ == '__main__':       
    app.run(debug=True)