import json
import os

import h5py
import numpy as np
import pandas as pd
import spacy
from keras.utils import to_categorical
from spacy.attrs import *
from tqdm import tqdm

from .DataSplitter import make_label_wise_train_test_split


class ETL:

    def __init__(self, config, tfhub):
        self.__dict__ = config
        self.nlp = spacy.load("en_core_web_sm")
        self.vec_dim = len(self.nlp(".")[0].vector)
        if self.google_vec:
            self.vec_dim += + 512
            self.tfhub = tfhub

    @staticmethod
    def get_citation_and_label(lines: list, vocab_x: set, vocab_y: set):
        citations = []
        labels = []
        tl = []
        tt = []
        for l in lines:
            arr = l.split(" ")
            # if arr[0].strip() in ["+L+", "+PAGE+", "\n"] and len(arr) > 1:
            #    arr[0] = "ÑŁ"
            if len(arr) == 1:
                citations.append(tt)
                labels.append(tl)
                tt = []
                tl = []
            else:
                if arr[len(arr) - 1] == "notes":
                    arr[len(arr) - 1] = "note"
                tt.append(arr[0])
                tl.append(arr[len(arr) - 1])
                vocab_x.add(arr[0])
                vocab_y.add(arr[len(arr) - 1])
        if len(tt) > 0:
            citations.append(tt)
            labels.append(tl)
        return citations, labels, vocab_x, vocab_y

    @staticmethod
    def parsecit_data_to_csv(data_dir_path, out_csv_path):

        vocabx = set()
        vocaby = set()
        data = []

        c_c = 0
        for f_name in os.listdir(data_dir_path):
            print(f"processing >> {os.path.join(data_dir_path, f_name)}")
            f = open(os.path.join(data_dir_path, f_name), 'r', encoding='utf-8')
            cont = f.read().split("\n")
            f.close()
            c, l, vocabx, vocaby = ETL.get_citation_and_label(lines=cont, vocab_x=vocabx, vocab_y=vocaby)
            for xs, ys in zip(c, l):
                ix = 0
                c_c = c_c + 1
                data.append({
                    "cit_id": c_c,
                    "text": xs,
                    "label": ys
                })
            print(f"Processed sequence {len(c)}")
        df = pd.DataFrame(data)
        df.to_csv(out_csv_path)

    @staticmethod
    def write_nparray(path, arr):
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("d", data=arr)
        hf.close()

    @staticmethod
    def read_nparray(path):
        arr = None
        with h5py.File(path, 'r') as hf:
            arr = hf['d'][:]
        hf.close()
        return arr

    def data_and_xydict(self, train_per=0.9):

        all_labs = None
        channel_dict = {"s1": [0], "s2": [0], "s3": [0]}
        file_name = self.start_file
        data_set_df_path = os.path.join(self.output_dir, file_name)
        lab_path = os.path.join(self.output_dir, "labels.json")
        channel_path = os.path.join(self.output_dir, "channels.json")

        if not os.path.exists(data_set_df_path):
            ETL.parsecit_data_to_csv(data_dir_path=self.input_dir, out_csv_path=data_set_df_path)

        train_x0_path = os.path.join(self.output_dir, "train_x0.h5")
        train_x1_path = os.path.join(self.output_dir, "train_x1.h5")
        train_y_path = os.path.join(self.output_dir, "train_y.h5")
        train_bio_path = os.path.join(self.output_dir, "train_bio.h5")
        test_x0_path = os.path.join(self.output_dir, "test_x0.h5")
        test_x1_path = os.path.join(self.output_dir, "test_x1.h5")
        test_y_path = os.path.join(self.output_dir, "test_y.h5")
        test_bio_path = os.path.join(self.output_dir, "test_bio.h5")

        if os.path.exists(train_x0_path) and \
                os.path.exists(train_x1_path) \
                and os.path.exists(train_y_path) \
                and os.path.exists(train_bio_path) \
                and os.path.exists(test_x0_path) \
                and os.path.exists(test_x1_path) \
                and os.path.exists(test_y_path) \
                and os.path.exists(test_bio_path):

            train_x0 = ETL.read_nparray(train_x0_path)
            train_x1 = ETL.read_nparray(train_x1_path)
            train_y = ETL.read_nparray(train_y_path)
            train_bio = ETL.read_nparray(train_bio_path)
            test_x0 = ETL.read_nparray(test_x0_path)
            test_x1 = ETL.read_nparray(test_x1_path)
            test_y = ETL.read_nparray(test_y_path)
            test_bio = ETL.read_nparray(test_bio_path)

            with open(lab_path) as f:
                all_labs = json.load(f)
            f.close()
            channel_dict
            with open(channel_path) as f:
                channel_dict = json.load(f)
            f.close()
            return {
                       "train": {
                           "x0": train_x0,
                           "x1": train_x1,
                           "y": train_y,
                           "bio": train_bio,
                       },
                       "test": {
                           "x0": test_x0,
                           "x1": test_x1,
                           "y": test_y,
                           "bio": test_bio,
                       },
                       "labels": all_labs
                   }, channel_dict
        else:
            df = pd.read_csv(data_set_df_path)
            ids = df.cit_id.tolist()
            labels = df.label.tolist()
            texts = df.text.tolist()

            all_labs = set()
            all_labs.add("<padd>")
            for l in labels:
                for e in eval(l):
                    all_labs.add(e)
            #all_labs.add("<nl>")
            all_labs = list(all_labs)
            all_labs.sort()
            all_labs = {x: i for i, x in enumerate(all_labs)}
            dummy_x = [0.] * self.vec_dim
            dummy_x = np.array(dummy_x)
            dummy_y = [0] * len(all_labs)
            dummy_y = np.array(dummy_y)
            with open(lab_path, 'w') as jf:
                json.dump(all_labs, jf)
            jf.close()

            train_df, test_df = make_label_wise_train_test_split(data_df=df, target_per=train_per)
            train_df.to_csv(os.path.join(self.output_dir, file_name.replace(".csv", "") + ".train.csv"))
            test_df.to_csv(os.path.join(self.output_dir, file_name.replace(".csv", "") + ".test.csv"))

            train_df_ids = train_df["cit_id"].tolist()

            train_x0 = []
            train_x1 = []
            train_y = []
            train_bio = []

            test_x0 = []
            test_x1 = []
            test_y = []
            test_bio = []

            max_len = -1

            count = 0

            s1 = channel_dict["s1"]
            s2 = channel_dict["s2"]
            s3 = channel_dict["s3"]

            print(" processing data ...")

            for cit, label in tqdm(zip(texts, labels)):
                cit = eval(cit)
                label = eval(label)

                doc = self.nlp(" ".join(cit))
                if len(doc) > max_len:
                    max_len = len(doc)

                spacyTokens = []
                spacyLabels = []
                spacyBIO = []

                i = 0
                for idx, act in enumerate(cit):
                    trg = doc[i].text
                    lb = all_labs[label[idx]]
                    if act == trg:
                        spacyTokens.append(doc[i])
                        spacyLabels.append(lb)
                        if i == 0:
                            spacyBIO.append([0, 1])
                        else:
                            spacyBIO.append([1, 0])
                    else:
                        spacyTokens.append(doc[i])
                        spacyLabels.append(lb)
                        if i == 0:
                            spacyBIO.append([0, 1])
                        else:
                            spacyBIO.append([1, 0])
                        while act != trg:
                            i = i + 1
                            spacyTokens.append(doc[i])
                            spacyLabels.append(lb)
                            if i == 0:
                                spacyBIO.append([0, 1])
                            else:
                                spacyBIO.append([1, 0])
                            trg += doc[i].text
                    i = i + 1

                spacy_x = np.array([x.vector for x in spacyTokens])
                if self.google_vec:
                    tfhub_x = self.tfhub.get_embedding([x.text for x in spacyTokens])
                    spacy_x = np.concatenate([tfhub_x, spacy_x], axis=-1)
                spacy_nlp = doc.to_array([POS, ENT_IOB, ENT_ID])
                t = np.ones_like(spacy_nlp)
                spacy_nlp = np.add(t, spacy_nlp)  # for padding - 0
                for el in set(spacy_nlp[:, 0:1].flatten().tolist()):
                    if el not in s1:
                        s1.append(el)
                for el in set(spacy_nlp[:, 1:2].flatten().tolist()):
                    if el not in s2:
                        s2.append(el)
                for el in set(spacy_nlp[:, 2:3].flatten().tolist()):
                    if el not in s3:
                        s3.append(el)
                spacy_y = to_categorical(spacyLabels, num_classes=len(all_labs))
                if ids[count] not in train_df_ids:
                    test_x0.append(spacy_x)
                    test_x1.append(spacy_nlp)
                    test_y.append(spacy_y)
                    test_bio.append(np.array(spacyBIO))
                else:
                    train_x0.append(spacy_x)
                    train_x1.append(spacy_nlp)
                    train_y.append(spacy_y)
                    train_bio.append(np.array(spacyBIO))
                count = count + 1

            print(f"\n Post Padding with MaxSeqLen-> {max_len}")

            for i in tqdm(range(len(train_x0))):
                x0 = train_x0[i]
                x1 = train_x1[i]
                y = train_y[i]
                bio = train_bio[i]
                rem = max_len - len(x0)
                for k in range(rem):
                    x0 = np.append(x0, [dummy_x], axis=0)
                    x1 = np.append(x1, [[0, 0, 0]], axis=0)
                    y = np.append(y, [dummy_y], axis=0)
                    bio = np.append(bio, [[0, 0]], axis=0)
                train_x0[i] = x0
                train_x1[i] = x1
                train_y[i] = y
                train_bio[i] = bio

            train_x0 = np.array(train_x0)
            train_x1 = np.array(train_x1)
            train_y = np.array(train_y)
            train_bio = np.reshape(train_bio, (len(train_bio), len(train_bio[0]), 2))

            for i in tqdm(range(len(test_x0))):
                x0 = test_x0[i]
                x1 = test_x1[i]
                y = test_y[i]
                bio = test_bio[i]
                rem = max_len - len(x0)
                for k in range(rem):
                    x0 = np.append(x0, [dummy_x], axis=0)
                    x1 = np.append(x1, [[0, 0, 0]], axis=0)
                    y = np.append(y, [dummy_y], axis=0)
                    bio = np.append(bio, [[0, 0]], axis=0)
                test_x0[i] = x0
                test_x1[i] = x1
                test_y[i] = y
                test_bio[i] = bio

            test_x0 = np.array(test_x0)
            test_x1 = np.array(test_x1)
            test_y = np.array(test_y)
            test_bio = np.reshape(test_bio, (len(test_bio), len(test_bio[0]), 2))

            ETL.write_nparray(train_x0_path, train_x0)
            ETL.write_nparray(train_x1_path, train_x1)
            ETL.write_nparray(train_y_path, train_y)
            ETL.write_nparray(train_bio_path, train_bio)

            ETL.write_nparray(test_x0_path, test_x0)
            ETL.write_nparray(test_x1_path, test_x1)
            ETL.write_nparray(test_y_path, test_y)
            ETL.write_nparray(test_bio_path, test_bio)

            with open(channel_path, 'w') as jf:
                json.dump(channel_dict, jf)
            jf.close()

            return {
                       "train": {
                           "x0": train_x0,
                           "x1": train_x1,
                           "y": train_y,
                           "bio": train_bio,
                       },
                       "test": {
                           "x0": test_x0,
                           "x1": test_x1,
                           "y": test_y,
                           "bio": test_bio,
                       },
                       "labels": all_labs
                   }, channel_dict
