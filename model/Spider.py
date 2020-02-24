import json
import urllib.request
import urllib

class Spider:
    def __init__(self, question):
        self.question = question
    def get_answer(self):
        data = {}
        data["appkey"] = "620e6c9fd799a995"
        data["question"] = self.question

        url_values = urllib.parse.urlencode(data)
        url = "https://api.binstd.com/iqa/query" + "?" + url_values

        result = urllib.request.urlopen(url)
        jsonarr = json.loads(result.read())

        if jsonarr["status"] != 0:
            print("msg", jsonarr["msg"])
            return ""

        result = jsonarr["result"]
        print("api content", result["content"])
        return result["content"]