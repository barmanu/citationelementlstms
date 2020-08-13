import json
import os
import random

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model
from spacy.attrs import *

seed_value = 123456
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class ParsCitLSTM:

    def __init__(self, model_config):
        self.__dict__ = model_config
        self.model = load_model(self.model_file)
        self.model.summary()
        self.idx2lab = self.load_json(self.label_dict_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.tfh_model = hub.load(self.tfhub_model_dir)
        self.vec_dim = len(self.nlp(".")[0].vector) + 512
        self.dummy_x = np.array([0.] * self.vec_dim)

    @staticmethod
    def load_json(path):
        d = None
        with open(path) as f:
            d = json.load(f)
        f.close()
        d = dict(zip(d.values(), d.keys()))
        return d

    @staticmethod
    def reshapex(seq, i, j):
        x = seq[:, :, i:j]
        x = np.reshape(x, (len(x), len(x[0])))
        return x

    def predict(self, text: str):
        text = text.replace("\n", "ÑŁ")
        doc = self.nlp(text)
        _t = [x.text for x in doc]
        _x = np.array([x.vector for x in doc])
        _ems = [np.array(x) for x in self.tfh_model(_t)]
        x = np.concatenate([_ems, _x], axis=-1)
        spacy_nlp = np.array(doc.to_array([POS, ENT_IOB, ENT_ID]))
        x = np.array([x])
        spacy_nlp = np.array([spacy_nlp])
        _y0, _y1 = self.model.predict(
            x = [
                x,
                self.reshapex(spacy_nlp, 0, 1),
                self.reshapex(spacy_nlp, 1, 2),
                self.reshapex(spacy_nlp, 2, 3)
            ]
        )
        return _y0, _y1


if __name__ == '__main__':



    tag = "/*/ce:bibliography/ce:bibliography-sec/ce:bib-reference/ce:other-ref"
    # for reproducibility

    df = pd.read_csv("../data/data.csv")#other_ref_citations_from_64k_data_set.csv")
    other_refs = df["text"].tolist()

    c = {
        "model_file": "../output/11:00:46/output/11:00:46/input_dim:608~hidden_dim:600~output_dim:14~lr:0.01~clip:5.0~beta1:0.9~beta2:0.999~l1:0.0~l2:0.0~drop_rate:0.05~batch:64~epoch:30~crf:False~rnn:True~num_of_rnn:1~s1:19~s2:5~s3:3.model-epoch-29.h5",
        "label_dict_file": "../data/labels.json",
        "tfhub_model_dir": "../resource/"
    }

    model = ParsCitLSTM(model_config=c)

    for text in other_refs[110:112]:
        text = " ".join(eval(text))
        y1, y2 = model.predict(text)
