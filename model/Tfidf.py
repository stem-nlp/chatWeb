from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from QA import QA

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

        print("tf_idf形状：",tf_idf.shape)
        return tf_idf.toarray()

    def get_feature(self, qa):
        """对待回答问题构建tf-idf特征"""
        text = [" ".join(qa.question_words)]
        X = self.vectorizer.transform(text)
        tf_idf = self.tf_idf_transformer.transform(X)
        return tf_idf.toarray()