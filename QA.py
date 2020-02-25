import jieba
import numpy as np
import os

dirname = os.path.dirname(__file__)
stop_words_path = os.path.join(dirname, "data/stop_word.txt")
with open(stop_words_path, "r" ,encoding='UTF-8') as f:
    stop_words = f.read().split('\n')

def cut_words(text):
    return [i for i in jieba.cut(text)]

def remove_stop_word(words):
    res = [i for i in words  if i not in stop_words]
    return res

class QA:
    def __init__(self, id, question, answer):
        self.id = id
        self.question = question
        self.answer = answer
        self.tokenize()
    def tokenize(self):
        self.question_words = cut_words(self.question)
        self.question_words_nostop = remove_stop_word(self.question_words)


if __name__ == '__main__':
    qa = QA(0,"此时有两种解决方案：","你好")
    qa.tokenize()