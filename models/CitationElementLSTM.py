import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import *
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix


class SimpleLSTMModel:

    def __init__(self, config_dict):
        self.__dict__ = config_dict
        tmp = []
        for k, v in config_dict.items():
            if k == "channel_dict":
                for m, n in v.items():
                    tmp.append(m + ":" + str(len(n) + 1))
            else:
                tmp.append(k + ":" + str(v))
        self.name = "~".join(tmp) + ".model"

    def get_model(self):
        """
        :return: model
        """

        inputs = Input(
            shape=(None, self.input_dim,),
        )

        masked = Masking(mask_value=0., input_shape=(None, self.input_dim,))

        channel_s1_input = Input(shape=(None,), name="s1-in")
        channel_s1_embedding = Embedding(
            200,
            max(int(max(self.channel_dict["s1"]) / 3), 5),
            mask_zero=True,
            input_length=None,
            name="s1"
        )(channel_s1_input)

        channel_s2_input = Input(shape=(None,), name="s2-in")

        channel_s2_embedding = Embedding(
            20,
            max(int(max(self.channel_dict["s2"]) / 3), 5),
            mask_zero=True,
            input_length=None,
            name="s2"
        )(channel_s2_input)

        channel_s3_input = Input(shape=(None,), name="s3-in")
        channel_s3_embedding = Embedding(
            10,
            max(int(max(self.channel_dict["s3"]) / 3), 5),
            mask_zero=True,
            input_length=None,
            name="s3"
        )(channel_s3_input)

        lstm = None
        if self.l2 == 0.0:
            lstm = Bidirectional(
                LSTM(
                    units=self.hidden_dim,
                    return_sequences=True,
                    name="lstm-0"
                ),
                merge_mode="concat",
                name="b_cat_rnn"
            )
        else:
            lstm = Bidirectional(
                LSTM(
                    units=self.hidden_dim,
                    kernel_regularizer=l2(self.l2),
                    bias_regularizer=l2(self.l2),
                    return_sequences=True,
                    name="lstm-0"
                ),
                merge_mode="concat",
                name="b_cat_rnn"
            )

        drop = Dropout(
            self.drop_rate,
            name="drop"
        )

        dense = TimeDistributed(
            Dense(
                self.output_dim,
                activation='softmax',
            ),
            name="output"
        )

        dense_bio = TimeDistributed(
            Dense(
                2,
                activation='softmax',
            ),
            name="bio"
        )

        crf = CRF(self.output_dim)

        optim = Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, amsgrad=False)

        x_cat = concatenate(
            [
                masked(inputs),
                channel_s1_embedding,
                channel_s2_embedding,
                channel_s3_embedding
            ],
            axis=-1,
        )

        x = None
        if self.rnn:
            x = drop(lstm(x_cat))
            rem = self.num_of_rnn - 1
            if rem > 0:
                for il in range(rem):
                    x = LSTM(
                        units=self.hidden_dim,
                        return_sequences=True,
                        name="lstm-" + str(il + 1)
                    )(x)

        else:
            x = x_cat

        if self.crf:

            outputs = crf(x)
            bio = dense_bio(x)

            model = Model(
                inputs=[inputs, channel_s1_input, channel_s2_input, channel_s3_input],
                outputs=[outputs, bio],
                name=self.name
            )

            model.compile(
                loss=crf_loss,
                optimizer=optim,
                metrics=crf_accuracy
            )
        else:
            outputs = dense(x)
            bio = dense_bio(x)
            model = Model(
                inputs=[inputs, channel_s1_input, channel_s2_input, channel_s3_input],
                outputs=[outputs, bio],
                name=self.name
            )
            model.compile(
                optimizer=optim,
                loss={
                    "output": "categorical_crossentropy",
                    "bio": "binary_crossentropy",
                },
                metrics={
                    "output": "categorical_accuracy",
                    "bio": "categorical_accuracy"
                },
            )
        model.summary()
        return model

    @staticmethod
    def np_array_2_cv_split(x0, x1, y, bio, split=5):
        sample_per_split = int(len(x0) / float(split))
        cv_dict = {}
        x0_splits = []
        x1_splits = []
        y_splits = []
        bio_splits = []
        start = 0
        for i in range(split - 1):
            x0_splits.append(x0[start:min((start + sample_per_split), len(x0)), :])
            x1_splits.append(x1[start:min((start + sample_per_split), len(x1)), :])
            y_splits.append(y[start:min((start + sample_per_split), len(y)), :])
            bio_splits.append(bio[start:min((start + sample_per_split), len(bio)), :])
            start += sample_per_split

        x0_splits.append(x0[start:len(x0), :])
        x1_splits.append(x1[start:len(x1), :])
        y_splits.append(y[start:len(y), :])
        bio_splits.append(bio[start:len(bio), :])

        for i in range(split):
            test_x0 = x0_splits[i]
            test_x1 = x1_splits[i]
            test_y = y_splits[i]
            test_bio = bio_splits[i]
            train_x0 = []
            train_x1 = []
            train_y = []
            train_bio = []
            for j in range(split):
                if i != j:
                    train_x0.append(x0_splits[j])
                    train_x1.append(x1_splits[j])
                    train_y.append(y_splits[j])
                    train_bio.append(bio_splits[j])
            train_x0 = np.concatenate(train_x0, axis=0)
            train_x1 = np.concatenate(train_x1, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            train_bio = np.concatenate(train_bio, axis=0)
            cv_dict[(i + 1)] = {
                "train_x0": np.array(train_x0),
                "train_x1": np.array(train_x1),
                "train_y": np.array(train_y),
                "train_bio": np.array(train_bio),
                "test_x0": np.array(test_x0),
                "test_x1": np.array(test_x1),
                "test_y": np.array(test_y),
                "test_bio": np.array(test_bio)
            }
        return cv_dict

    def cross_validate(self, x0: np.array, x1: np.array, y: np.array, bio: np.array, batch: int, epoch: int,
                       label_dict: dict, split=5,
                       verbose=2):
        cv_dict = SimpleLSTMModel.np_array_2_cv_split(x0=x0, x1=x1, y=y, bio=bio, split=split)
        cv_scores_and_models = []
        cls_report = []
        fc = 1
        for k, v in cv_dict.items():
            model = self.get_model()
            hist, model, _ = self.train_model(
                model,
                train_data={
                    "x0": v["train_x0"],
                    "x1": v["train_x1"],
                    "y": v["train_y"],
                    "bio": v["train_bio"]
                },
                test_data={
                    "x0": v["test_x0"],
                    "x1": v["test_x1"],
                    "y": v["test_y"],
                    "bio": v["test_bio"]
                },
                label_dict=label_dict,
                batch=batch,
                epoch=epoch,
                verbose=verbose
            )
            hist["fold"] = fc
            fc = fc + 1
            score_list = hist["score"].tolist()
            last_score = score_list[len(score_list) - 1]
            cls_report.append(last_score)
            cv_scores_and_models.append((last_score, model))
        return cv_scores_and_models, cls_report

    @staticmethod
    def reshapex(seq, i, j):
        x = seq[:, :, i:j]
        x = np.reshape(x, (len(x), len(x[0])))
        return x

    def train_model(self, model: Model, train_data: dict, test_data: dict, label_dict: dict, batch: int, epoch: int,
                    verbose: int):
        train_history = []
        model_store = []
        for i in range(epoch):
            model.fit(
                x=[train_data["x0"], self.reshapex(train_data["x1"], 0, 1), self.reshapex(train_data["x1"], 1, 2),
                   self.reshapex(train_data["x1"], 2, 3)], y=[train_data["y"], train_data["bio"]],
                validation_data=(
                    [test_data["x0"], self.reshapex(test_data["x1"], 0, 1), self.reshapex(test_data["x1"], 1, 2),
                     self.reshapex(test_data["x1"], 2, 3)], [test_data["y"], test_data["bio"]]),
                batch_size=batch,
                epochs=1,
                verbose=verbose
            )
            f1, rdict = self.eval_model(model=model, eval_data=test_data, label_dict=label_dict)
            print(f" epoch:{i + 1} >===============> Macro-F1:{f1}  SER:{rdict['ser']} JER:{rdict['ser']}")
            rdict["epoch"] = (i + 1)
            rdict["score"] = f1

            train_history.append(rdict)
            model_store.append(model)
        train_history = pd.DataFrame(train_history)
        return train_history, model, model_store

    @staticmethod
    def calulate_ser_jer(y_true, y_pred, keep_tag):
        yt = [1 if x == keep_tag else 0 for x in y_true]
        yp = [1 if x == keep_tag else 0 for x in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_true=yt, y_pred=yp).ravel()
        ser = 0.0
        if (tp + fp) > 0.0:
            ser = fp / float(tp + fp)
        jer = 0.0
        if (tp + fn) > 0.0:
            jer = fn / float(tp + fn)
        return ser, jer

    def eval_model(self, model: Model, eval_data: dict, label_dict: dict):
        gold_y = eval_data["y"]
        gold_bio = eval_data["bio"]
        gold_y = np.argmax(gold_y, axis=-1).flatten()
        gold_bio = np.argmax(gold_bio, axis=-1).flatten()
        xin = [
            eval_data["x0"],
            self.reshapex(eval_data["x1"], 0, 1),
            self.reshapex(eval_data["x1"], 1, 2),
            self.reshapex(eval_data["x1"], 2, 3)
        ]
        pred_y, pred_bio = model.predict(x=xin)
        pred_y = np.argmax(pred_y, axis=-1).flatten()
        pred_bio = np.argmax(pred_bio, axis=-1).flatten()
        g, p = [], []
        gbio, pbio = [], []
        for i in range(len(gold_y)):
            if label_dict[gold_y[i]] != "<padd>":
                g.append(gold_y[i])
                p.append(pred_y[i])
                gbio.append(gold_bio[i])
                pbio.append(pred_bio[i])
        f1 = f1_score(g, p, average="macro")
        rdict = {}
        print(classification_report(g, p))
        d = classification_report(g, p, output_dict=True)
        for sk, sv in d.items():
            if sk in ["macro avg", "weighted avg"]:
                rdict[sk.replace(" ", "_")] = sv["f1-score"]
            elif sk == "accuracy":
                rdict["accuracy"] = sv
            elif int(sk) in label_dict.keys():
                rdict[label_dict[int(sk)]] = sv["f1-score"]
        ser, jer = self.calulate_ser_jer(gbio, pbio, 1)
        rdict["ser"] = ser
        rdict["jer"] = jer
        return f1, rdict
