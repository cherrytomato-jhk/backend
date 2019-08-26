import numpy as np
import tensorflow as tf
import json
import nrekit
from collections import defaultdict
from rake_nltk import Rake
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

rake = Rake()

class RelationExtractor():
    def __init__(self):
        # load data files
        self._load_data()

        # setting model
        self._set_model()

    def _load_data(self):
        print("loading data...")
        self.rel2id = json.load(open("./data/rel2id.json"))
        self.word2id = json.load(open("./data/word_vec_word2id.json"))
        self.word_vec = np.load("./data/word_vec_mat.npy")

        self.id2rel = dict((y,x) for x,y in self.rel2id.items())

        self.rel_tot = len(self.rel2id)
        self.word_tot = len(self.word_vec)
        print("loading data complete...")
    
    def _set_model(self):
        self.max_length = 120
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='word')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name="mask")
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='pos2')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[1, 2], name='scope')
        
        self._model()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        dataset_name = "nyt"
        encoder = "pcnn"
        selector = "att"
        self.saver = self.saver.restore(self.sess, "./checkpoint/" + dataset_name + "_" + encoder + "_" + selector)

    def _preprocess_data(self, data):


        sentences = sent_tokenize(data.lower())
        print(sentences)
        sentence_size = len(sentences)

        sentence_words = []
        sentence_word_vec = []

        mask = np.zeros([sentence_size, self.max_length], dtype=np.int32)
        pos1 = np.zeros([sentence_size, self.max_length], dtype=np.int32)
        pos2 = np.zeros([sentence_size, self.max_length], dtype=np.int32)
        length = np.full([sentence_size], self.max_length)
        scope = np.zeros([sentence_size, 1, 2], dtype=np.int32)

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) <= 0 :
                pass
            
            words = sentence.strip().split()
            word_vec = []
            for word in words:
                try:
                    word_vec.append(self.word2id[word])
                except:
                    word_vec.append(self.word2id['UNK'])
            if len(word_vec) < self.max_length:
                length[i] = len(word)
                for j in range(len(word_vec), self.max_length):
                    word_vec.append(self.word2id['BLANK'])
            else:
                word_vec[-1] = word_vec[-1][:self.max_length]

            sentence_words.append(words)
            sentence_word_vec.append(word_vec)

        key1_list = []
        key2_list = []

        for i in range(sentence_size):
            rake.extract_keywords_from_text(sentences[i])
            key1, key2 = rake.get_ranked_phrases()[:2]
            key1_list.append(key1)
            key2_list.append(key2)
            p1 = sentence_words[i].index(key1.lower())
            p2 = sentence_words[i].index(key2.lower())
            for j in range(self.max_length):
                pos1[i][j] = j - p1 + self.max_length
                pos2[i][j] = j - p2 + self.max_length
                if j >= self.max_length:
                    mask[i][j] = 0
                elif j <= min(p1, p2):
                    mask[i][j] = 1
                elif j <= max(p1, p2):
                    mask[i][j] = 2
                else:
                    mask[i][j] = 3
            scope[i][0] = [i-1, i]
        
        data = {
            'instances' : sentence_size,
            'words' : sentence_word_vec,
            'mask' : mask,
            'pos1' : pos1,
            'pos2' : pos2,
            'length' : length,
            'scope' : scope,
            'key1' : key1_list,
            'key2' : key2_list
        }
        return data
    
    def _model(self):
        self.x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec, self.pos1, self.pos2)
        self.x_test = nrekit.network.encoder.pcnn(self.x, self.mask, keep_prob=1.0)
        self.test_logit, self.test_repre = nrekit.network.selector.bag_attention(self.x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
        # self._output = tf.cast(tf.argmax(test_logit, -1), tf.int32) # only necessary when need only max arg
    
    def get_result(self, data, topN=3):
        preprocessed = self._preprocess_data(data)
        logit = []
        for i in range(preprocessed['instances']):
            score = self.sess.run([self.test_logit], feed_dict={
                                                    self.mask: preprocessed['mask'],
                                                    self.word: preprocessed['words'],
                                                    self.pos1: preprocessed['pos1'],
                                                    self.pos2: preprocessed['pos2'],
                                                    self.length: preprocessed['length'],
                                                    self.scope: preprocessed['scope'][i]
                                                })
            logit.extend(score[0])
        topNresult = defaultdict(list)
        for idx, scores in enumerate(logit):
            topNidx = np.argsort(scores)[-topN:]
            topNrel = []
            for idx in topNidx:
                topNrel.append(self.id2rel[idx])
            topNvalues = [scores[i] for i in topNidx]
            topNresult[idx].append({x:y for x,y in zip(topNrel, topNvalues)})
        return self._to_dict(topNresult, preprocessed['key1'], preprocessed['key2'], preprocessed['instances'])
    
    def _to_dict(self, result, key1, key2, instances, topN=3):
        results = []
        for i in range(instances):
            summary = {}
            for rel in result[i]:
                score_str = ""
                for k, v in rel.items():
                    score_str += "{}({})\n".format(k, v)
            summary = {'head': key1[i], 'tail': key2[i], 'rel': score_str}
            results.append(summary)
        return results
        
if __name__ == "__main__":
    relExtractor = RelationExtractor()
    corpus = "Bill_Gates is the founder of Microsoft . Bill_Gates is the founder of Microsoft . Bill_Gates is the founder of Microsoft".lower()
    results = relExtractor.get_result(corpus)
    print(results)
