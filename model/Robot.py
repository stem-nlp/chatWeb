import xlrd
import numpy as np
from QA import QA, cut_words
from model.Spider import Spider
from model.Bert import Bert
from model.Tfidf import TFIDF
from model.BoolSearch import BoolSearch
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from config import *

import os, sys
import pickle

xls_path = os.path.join(os.path.dirname(__file__), "../data/qa_corpus.xlsx")
category_save_path = os.path.join(os.path.dirname(__file__), "../save/categories.pkl")
cluster_save_path = os.path.join(os.path.dirname(__file__), "../save/cluster.m")
feature_save_path = os.path.join(os.path.dirname(__file__), "../save/feature.npy")
cluster_csv_save_path = os.path.join(os.path.dirname(__file__), "../save/cluster_output.csv")


def read_data():
    workbook = xlrd.open_workbook(xls_path)
    sheet = workbook.sheet_by_index(0)

    qa_list = []  # 包含所有“问题-答案”对象的列表
    for i in range(2, sheet.nrows):
        qid = sheet.cell(i, 0).value
        question = str(sheet.cell(i, 1).value)
        answer = str(sheet.cell(i, 2).value)
        qa_list.append(QA(qid, question, answer))
    return qa_list


class Robot():
    def __init__(self):
        # 是否重新训练聚类器
        retrain = False
        # 获取训练数据，构造特征，获得特征矩阵
        print("读取语料数据")
        self.qa_list = read_data()

        # 加载布尔搜索
        print("构建布尔搜索索引")
        self.bool_search = BoolSearch(self.qa_list)

        # 选择特征提取方法：TFIDF Bert
        self.feature_extractor_name = "BERT"
        if self.feature_extractor_name == "TFIDF":
            self.feature_extractor =  TFIDF()
        elif self.feature_extractor_name == "BERT":
            self.feature_extractor = Bert()

        # 提取语料库特征
        self.feature_matrix = None
        if not os.path.exists(feature_save_path) or self.feature_extractor_name == "TFIDF" or retrain:
            print("提取语料库特征")
            self.feature_matrix = self.feature_extractor.word_feature(self.qa_list)
            if self.feature_extractor_name != "TFIDF":
                np.save(feature_save_path, self.feature_matrix)
        else:
            self.feature_matrix = np.load(feature_save_path)


        # 训练聚类器, 聚类结果
        self.categories = None
        self.cls = None
        if not os.path.exists(cluster_save_path) or not os.path.exists(category_save_path) or retrain:
            print("训练聚类器，获得语料库聚类结果")
            self.categories, self.cls = self.cluster_train(self.feature_matrix)

            # 保存
            with open(category_save_path, 'wb') as f:
                pickle.dump(self.categories, f)
            joblib.dump(self.cls, cluster_save_path)

            # 查看类别
            with open(cluster_csv_save_path, 'w', encoding='utf8') as f:
                for idx, q in enumerate(self.qa_list):
                    f.write(q.question + ',' + str(self.categories[idx]) + '\n')
        else:
            with open(category_save_path, 'rb') as f:
                self.categories = pickle.load(f)
            self.cls = joblib.load(cluster_save_path)

    def ask(self, question):
        answer = ""
        # 预测
        qa = QA(None, question, "")
        feature = self.feature_extractor.get_feature(qa)
        # 查询语句所属类别
        c = self.cluster_pred(feature, self.cls, thresh=SPIDER_THRESHOLD)

        # 类别为-1，表示问题与聚类簇距离过大，应使用其他方法
        if c != -1:
            # 取出该类别的所有问答对的id
            find_items = np.where(self.categories == c)[0]
            if len(find_items) == 0:
                return ERROR_REPLY
            # 获取特征向量
            find_feature = self.feature_matrix[find_items]
            print("找到{}个候选问题，计算相似度...".format(len(find_feature)))

            # 计算各向量之间的余弦相似度
            rank_list = cosine_similarity(feature, find_feature)[0]
            # 找出前top_k个最相似句子
            top_k = 10
            top_arg = np.argsort(rank_list)[::-1][:top_k]

            # 基于特征的最相似问题topk结果：[[qa, 相似度], [qa, 相似度], [qa, 相似度]...]
            feature_based_result = []
            for i in top_arg:
                feature_based_result.append([self.qa_list[find_items[i]], rank_list[i]])

            # 获取布尔搜索topk结果
            bs_based_result = self.bool_search.search(qa, top_k, scale=BOOL_SEARCH_SCALE)

            # 结果合并，排序
            total_result = feature_based_result + bs_based_result

            if DEBUG:
                for cnt, item in enumerate(total_result):
                    total_result[cnt].append("特征" if cnt<top_k else "布尔搜索")

            sorted(total_result, key=lambda x:x[1])

            if DEBUG:
                for cnt, (q, rate, type_) in enumerate(total_result):
                    print(q.question, rate, type_)

            max_qa = total_result[0][0]

            print("匹配问题：{}\n对应回答：{}".format(max_qa.question, max_qa.answer))
            answer = max_qa.answer
        else:
            # 爬虫
            sp = Spider(question)
            sp_res = sp.get_answer()

            # 爬虫返回空串，则尝试使用生成模型
            if sp_res in ["", "defaultReply"]:
                # 生成模型
                answer = UNKNOWN_REPLY
            else:
                answer = sp_res
        return answer

    def cluster_train(self, feature_matrix):
        '''
        聚类
        '''
        cluster_ = KMeans(n_clusters=12, random_state=9)
        category = cluster_.fit_predict(feature_matrix)
        print(category.shape)
        return category, cluster_

    def cluster_pred(self, x, cluster_, thresh=0.5):
        '''
        判断x所属聚类的类别，若x距离该类别距离大于阈值thresh, 则返回-1
        :param x:
        :param cluster_:
        :return:
        '''
        c = cluster_.predict(x)
        center = cluster_.cluster_centers_[c]
        distance = euclidean_distances(center, x)
        print("最近类别：{}，最近欧拉距离：{}".format(c, distance))
        return c if distance[0][0] < thresh else -1


if __name__ == '__main__':
    r = Robot()
    r.ask("我要现金存款")
