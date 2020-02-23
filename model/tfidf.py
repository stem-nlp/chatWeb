import xlrd
import numpy as np
from QA import QA, cut_words
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os, sys
import pickle

xls_path = os.path.join( sys.path[0], "data/qa_corpus.xlsx")
category_save_path = os.path.join( sys.path[0], "save/categories.pkl")
cluster_save_path = os.path.join( sys.path[0], "save/cluster.m")

def read_data():
    workbook = xlrd.open_workbook(xls_path)
    sheet = workbook.sheet_by_index(0)

    qa_list = [] # 包含所有“问题-答案”对象的列表
    for i in range(2, sheet.nrows):
        qid = sheet.cell(i, 0).value
        question = str(sheet.cell(i, 1).value)
        answer = str(sheet.cell(i, 2).value)
        qa_list.append(QA(qid, question, answer))
    return qa_list

class TFIDF:
    def __init__(self):
        pass

    def word_feature(self, qa_list: [QA]):
        """对语料库构建tf-idf特征"""
        corpus = []
        for i in qa_list:
            corpus.append(" ".join(i.question_words))

        self.vectorizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
        self.vectorizer.fit(corpus)
        X = self.vectorizer.transform(corpus)

        self.tf_idf_transformer = TfidfTransformer()
        self.tf_idf_transformer.fit(X)
        tf_idf = self.tf_idf_transformer.transform(X)

        print(tf_idf.shape)
        return tf_idf.toarray()

    def get_feature(self, qa):
        """对待回答问题构建tf-idf特征"""
        text = [" ".join(qa.question_words)]
        X = self.vectorizer.transform(text)
        tf_idf = self.tf_idf_transformer.transform(X)
        return tf_idf.toarray()



class Robot():
    def __init__(self):
        # 获取训练数据，构造特征，获得特征矩阵
        self.qa_list = read_data()
        self.tfidf = TFIDF()
        self.feature_matrix = self.tfidf.word_feature(self.qa_list)

        # 训练聚类器
        self.cls = None
        if not os.path.exists(cluster_save_path):
            print("找不到分类器，重新训练")
            self.cls = self.cluster_train(self.feature_matrix)
            joblib.dump(self.cls, cluster_save_path)
        else:
            self.cls = joblib.load(cluster_save_path)

        # 获取聚类结果
        self.categories = None
        if not os.path.exists(category_save_path):
            print("找不到已聚类结果，重新聚类")
            self.categories = self.cluster_pred(self.feature_matrix, self.cls)
            with open(category_save_path, 'wb') as f:
                pickle.dump(self.categories, f)
        else:
            with open(category_save_path, 'rb') as f:
                self.categories = pickle.load(f)

    def ask(self, question):
        # 预测
        qa = QA(None, question ,"")
        feature = self.tfidf.get_feature(qa)
        # 查询语句所属类别
        c = self.cluster_pred(feature, self.cls)

        # 取出该类别的所有问答对的id
        find_items = np.where(self.categories == c)
        if len(find_items[0]) == 0:
            return False
        # 获取特征向量
        find_feature = self.feature_matrix[find_items[0]]

        # 计算各向量之间的余弦相似度
        print("找到{}个候选问题，计算相似度...".format(len(find_feature)))
        cs = cosine_similarity(feature, find_feature)
        # 取查询句子相对其他句子的相似度，找出最相似句子对应答案
        rank_list = cs[0]
        max_qa_index = find_items[0][np.argmax(rank_list)]
        max_qa = self.qa_list[max_qa_index]

        print("匹配问题：{}\n对应回答：{}".format(max_qa.question, max_qa.answer))
        return max_qa.answer

    def cluster_train(self, feature_matrix):
        '''
        聚类
        '''
        cluster_ = KMeans(n_clusters=8, random_state=9)
        cluster_.fit(feature_matrix)
        return cluster_

    def cluster_pred(self, x, cluster_):
        c = cluster_.predict(x)
        return c

if __name__ == '__main__':
    r = Robot()
    r.ask("我要现金存款")