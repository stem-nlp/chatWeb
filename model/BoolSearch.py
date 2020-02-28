import os
import collections

class BoolSearch:
    def __init__(self, qa_list):
        self.qa_list = qa_list
        self.index_table = {}
        for idx, qa in enumerate(self.qa_list):
            for word in list(set(qa.question_words_nostop)):
                if word not in self.index_table.keys():
                    self.index_table[word] = []
                self.index_table[word].append(idx)

    def search(self, qa, top_k=10, scale=1.0):
        '''
        布尔搜索
        :param qa: QA实例
        :param top_k: 返回前top_k个最相似qa
        :param scale: 匹配度系数
        :return: 前top_k个最相似qa，以及qa的问题中包含查询句子单词的数量占比
                [[qa, count/total_words_cnt], [qa, count/total_words_cnt], ...]
        '''
        query_counts = collections.Counter()
        # 查找每个单词在哪些问题中出现过
        total_words_cnt = len(qa.question_words_nostop)
        for w in qa.question_words_nostop:
            query_res = self.index_table.get(w, [])
            query_counts.update(query_res)
        # 找出现最高的topk个问题id
        query_result = query_counts.most_common(top_k)
        qa_result = [[self.qa_list[qid], (cnt * scale)/total_words_cnt] for qid, cnt in query_result]

        return qa_result

if __name__ == '__main__':
    import xlrd
    xls_path = os.path.join(os.path.dirname(__file__), "../data/qa_corpus.xlsx")
    from QA import QA
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
    qa_list = read_data()
    bs = BoolSearch(qa_list)
    qa_result = bs.search(QA("", "咨询一下如何存款", ""))
    for qa in qa_result:
        print("{}:{}".format(qa.question, qa.answer))