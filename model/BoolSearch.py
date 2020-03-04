import os
import collections

class BoolSearch:
    def __init__(self, qa_list):
        self.qa_list = qa_list
        self.index_table = {}#{"word a":[ida1,ida2,ida3..."word b":[idb1,idb2,idb3...]}
        for idx, qa in enumerate(self.qa_list):
            #qa_cutted=list(set(qa.question_words_nostop))
            qa_cutted=list(set(qa.question_words))# ipfgao更新：对于现有库，保留stop word 效果较佳，或者重新优化stop word列表

            #避免在去掉stop word后出现空值
            if len(qa_cutted)==0:qa_cutted=list(set(qa.question_words))
                
            for word in qa_cutted:
                if word not in self.index_table.keys():
                    self.index_table[word] = []
                self.index_table[word].append((idx,len(qa_cutted)))## ipfgao更新：记录qa的长度，方便提出的问题跟库里的问题进行对比

    def search(self, qa, top_k=10, scale=1.0):
        '''
        布尔搜索
        :param qa: QA实例
        :param top_k: 返回前top_k个最相似qa
        :param scale: 匹配度系数
        :return: 前top_k个最相似qa，以及qa的问题中包含查询句子单词的数量占比
                [[qa, count**2/(total_words_cnt*len(qa))], [qa, count**2/(total_words_cnt*len(qa))], ...]
        '''
        query_counts = collections.Counter()#计数器
        # 查找每个单词在哪些问题中出现过
        #qa_cutted=qa.question_words_nostop
        qa_cutted=qa.question_words# ipfgao更新：对于现有库，保留stop word 效果较佳，或者重新优化stop word列表
        total_words_cnt = len(qa_cutted)#总单词数
        
        #避免在去掉stop word后出现空值
        if total_words_cnt==0:
            qa_cutted=qa.question_words
            total_words_cnt=len(qa_cutted)
            
        print(f"分词后的问题：{qa_cutted}")    
        
        for w in qa_cutted:
            query_res = self.index_table.get(w, [])
            query_counts.update(query_res)

        #ipfgao更新：同频词太多，直接返回topk并不合理，应该返回最大的那个（或者那一群）,然后选取top_k个
        max_cnt=max(query_counts.values())
        # ipfgao更新：以 提出问题 跟 库中问题 共有词汇 在二者中的占比之积 做为二者的匹配度
        qa_result = [[self.qa_list[qid[0]], (max_cnt * scale)**2/(total_words_cnt*qid[1])] for qid,cnt in query_counts.items() if cnt==max_cnt]
        qa_result=sorted(qa_result, key=lambda x:x[1],reverse=True)[0:top_k]
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
    qa_test = QA("", "咨询一下如何存款", "")
    qa_result = bs.search(qa_test)
    for qa in qa_result:
        print("{}:{},\n 匹配分值：{}".format(qa[0].question, qa[0].answer, qa[1]))
