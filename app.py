from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import cross_origin
import json

from model import tfidf
app = Flask(__name__, static_folder='dist', template_folder="dist", static_url_path='')

robot = tfidf.Robot()
@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")

@app.route('/api/ask', methods=["POST"])
@cross_origin()
def model():
    try:
        raw_data = request.get_data()
        json_data = json.loads(raw_data)
        input_question = json_data.get('question','')
        print("用户输入：{}".format(input_question))

        answer = robot.ask(input_question)
        if not answer:
            answer = "抱歉，我还不能回答你的问题"
        result = {
            "code": 0,
            "data": {
                "content": answer
            }
        }
        return jsonify(result)
    except Exception as e:
        print(str(e))
        return jsonify({"code":-1, "data":str(e)})


if __name__ == '__main__':
    app.run()

application = app