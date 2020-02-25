import torch
from QA import QA
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import math


class Bert:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')
    def word_feature(self, qa_list: [QA]):
        corpus = []
        for i in qa_list:
            corpus.append("".join(i.question_words_nostop))
        feature_matrix = self.encode(corpus)
        return feature_matrix.cpu().numpy()

    def get_feature(self, qa):
        text = qa.question_words_nostop
        return self.encode(["".join(text)]).cpu().numpy()

    def encode(self, text_list):
        encoded_dict = self.tokenizer.batch_encode_plus(text_list,
                                                          max_length=30,
                                                          pad_to_max_length=True)
        indexed_tokens = encoded_dict["input_ids"]
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        # Load pre-trained model (weights)
        self.model.eval()

        # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # self.model.to('cuda')

        # Predict hidden states features for each layer
        encode_result = []
        batch_size = 256
        iter_num =  math.ceil(tokens_tensor.shape[0] / batch_size)
        with torch.no_grad():
            for i in range(iter_num):
                print("{}/{}".format(i, iter_num))
                # See the models docstrings for the detail of the inputs
                outputs = self.model(tokens_tensor[i*batch_size:(i+1)*batch_size])
                # In our case, the first element is the hidden state of the last layer of the Bert model
                encoded_layers = outputs[0]
                encode_result.append(torch.max(encoded_layers, dim=1)[0])

        encode_result = torch.cat(encode_result, dim=0)
        # return the 1st position——[CLS] hidden state
        return encode_result


if __name__ == '__main__':
    q1 = QA(0, "你好，机器人","1")
    q2 = QA(1, "明天天气怎么样","1")
    q3 = QA(2, "我要存钱","1")

    b = Bert()
    fm = b.word_feature([q1, q2, q3])
    # fm = b.get_feature(q1)
    print(fm.shape)