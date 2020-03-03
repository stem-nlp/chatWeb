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
import time
import heapq
import random


#xls_path = os.path.join( sys.path[0], "data/qa_corpus.xlsx")
#category_save_path = os.path.join( sys.path[0], "save/categories.pkl")
#cluster_save_path = os.path.join( sys.path[0], "save/cluster.m")
path="/root/private/chatWeb"
xls_path = os.path.join( path, "data/qa_corpus.xlsx")
category_save_path = os.path.join( path, "save/categories.pkl")
cluster_save_path = os.path.join( path, "save/cluster.m")
cluster_save_path_0 = os.path.join( path, "save/cluster_0.m")
category_save_path_0 = os.path.join( path, "save/categories_0.pkl")


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
        feature_matrix = self.tfidf.word_feature(self.qa_list)

        # 训练聚类器
        self.cls = None
        if not os.path.exists(cluster_save_path):
            print("找不到分类器，重新训练")
            self.cls = self.cluster_train(feature_matrix)
            joblib.dump(self.cls, cluster_save_path)
        else:
            self.cls = joblib.load(cluster_save_path)

        # 获取聚类结果
        self.categories = None
        if not os.path.exists(category_save_path):
            print("找不到已聚类结果，重新聚类")
            self.categories = self.cluster_pred(feature_matrix, self.cls)
            with open(category_save_path, 'wb') as f:
                pickle.dump(self.categories, f)
        else:
            with open(category_save_path, 'rb') as f:
                self.categories = pickle.load(f)
        
        #对聚类中结果二次聚类
        feature_matrix_0=feature_matrix[np.where(self.categories == 0)[0]]
        # 训练二次聚类器
        self.cls_0 = None
        if not os.path.exists(cluster_save_path_0):
            print("找不到二次分类器，重新训练")
            self.cls_0 = self.cluster_train(feature_matrix_0)
            joblib.dump(self.cls_0, cluster_save_path_0)
        else:
            self.cls_0 = joblib.load(cluster_save_path_0)

        # 获取二次聚类结果
        self.categories_0 = None
        if not os.path.exists(category_save_path_0):
            print("找不到二次已聚类结果，重新聚类")
            self.categories_0 = self.cluster_pred(feature_matrix_0, self.cls_0)
            with open(category_save_path_0, 'wb') as f:
                pickle.dump(self.categories_0, f)
        else:
            with open(category_save_path_0, 'rb') as f:
                self.categories_0 = pickle.load(f)


        
        
        # 建立特征向量词典
        self.feature=dict()#self.feature={0:{0:matrix,1:matrix....},1:matrix,2:matrix....}
        for c in range(10):        
            if c==0:
                feature_0=dict()
                for c_0 in range(10):
                    c_0_index = np.where(self.categories_0 == c_0)#c_0类的在C类行号
                    feature_0[c_0] = feature_matrix_0[c_0_index[0]]#行号在feature_matrix_0中对应的矩阵
                self.feature[c] = feature_0
            else:
                c_index = np.where(self.categories == c)
                self.feature[c] = feature_matrix[c_index[0]]

    def ask(self, question):
        # 预测
        qa = QA(None, question ,"")
        feature = self.tfidf.get_feature(qa)
        # 查询语句所属类别
        c = self.cluster_pred(feature, self.cls)
        
        # 提取出c类的特征向量
        c_feature = self.feature[int(c)]
        c_index = np.where(self.categories == c)[0]
        if c==0:
            c_0=self.cluster_pred(feature, self.cls_0)
            c_0_index = np.where(self.categories_0 == c_0)[0]     #c_0类的在c类中的行号
            c_index = c_index[c_0_index]                       #c_0类的在全类中的真实行号
            c_feature = c_feature[int(c_0)]


        # 计算各向量之间的余弦相似度
        print("找到{}个候选问题，计算相似度...".format(len(c_feature)))
        
        start_time = time.time()
        cs = cosine_similarity(feature, c_feature)
        end_time = time.time()
        print(f'cosine_similarity消耗时间：{end_time - start_time}')
        
        
        # 取查询句子相对其他句子的相似度，找出最相似句子对应答案
        # 随机选择前10个中的一个
        rank_list = list(cs[0])
        rank_index=list(map(rank_list.index, heapq.nlargest(5, rank_list)))
        answer_qa_index = c_index[random.choice(rank_index)]
        answer_qa = self.qa_list[answer_qa_index]

        print("匹配问题：{}\n对应回答：{}".format(answer_qa.question, answer_qa.answer))
        return answer_qa.answer

    def cluster_train(self, feature_matrix):
        '''
        聚类
        '''
        cluster_ = KMeans(n_clusters=10, random_state=0)
        cluster_.fit(feature_matrix)
        return cluster_

    def cluster_pred(self, x, cluster_):
        c = cluster_.predict(x)
        return c

if __name__ == '__main__':
    r = Robot()
    r.ask("我要现金存款")

#其他备注：
#从获取到id中随机获取5000个np.random.choice(find_items[0],5000,replace=False)
