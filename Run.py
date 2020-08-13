import argparse
import json
import os
import pprint
import random
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from etl.ETL import ETL
from models.CitationElementLSTM import SimpleLSTMModel

VERBOSE = 2


def load_json(path):
    config = None
    with open(path) as f:
        config = json.load(f)
    f.close()
    return config


class TFHubModel:

    def __init__(self, path):
        self.model = hub.load(path)
        self.dim = 512

    def get_embedding(self, text: list):
        ems = [np.array(x) for x in self.model(text)]
        return ems


if __name__ == '__main__':

    # for reproducibility
    seed_value = 123456
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--data-config",
        dest="data_config_path",
        default="config/data_config.json",
        help="path for data config json"
    )

    parser.add_argument(
        "--model-config",
        dest="model_config_path",
        default="config/model_config.json",
        help="path for model config"
    )

    parser.add_argument(
        "--cross-validate",
        dest="cross_validate",
        default=False,
        help="cross_validate or train_test"
    )

    parser.add_argument(
        "--fold",
        dest="fold",
        default=10,
        help="num of cross validation fold"
    )

    parser.add_argument(
        "--train-per",
        dest="train_per",
        default=0.8,
        help="% of train data when doing train/test"
    )

    parser.add_argument(
        "--output",
        dest="output",
        default="output",
        help="output dir for models and results"
    )

    parser.add_argument(
        "--tfhub-model-dir",
        dest="tfhub_model_dir",
        default="./resource/",
        help="resource dir for tfhub model"
    )

    args = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)
    print("Options given:\n")
    for k, v in vars(args).items():
        print(f"--{k} {v}")
    print("\n")

    t = time.localtime()
    args.output = os.path.join(args.output, str(time.strftime("%H:%M:%S", t)))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ems = TFHubModel(args.tfhub_model_dir)
    etl = ETL(config=load_json(args.data_config_path), tfhub=ems)
    data_dict, channel_dict = etl.data_and_xydict(train_per=args.train_per)
    train = data_dict["train"]
    test = data_dict["test"]
    lab_dict = data_dict["labels"]
    lab_dict = dict(zip(lab_dict.values(), lab_dict.keys()))

    model_config = dict(load_json(args.model_config_path))
    # adding input and output dim to data
    model_config["input_dim"] = etl.vec_dim
    model_config["output_dim"] = len(lab_dict)
    model_config["channel_dict"] = channel_dict

    pp = pprint.PrettyPrinter(indent=4)
    print("Options given:\n")
    for k, v in model_config.items():
        print(f"--{k} {v}")
    print("\n")

    model = None
    # training/cv
    CEL = SimpleLSTMModel(config_dict=model_config)
    if args.cross_validate:
        cv_scores_and_models, report = CEL.cross_validate(
            x0=np.concatenate([train["x0"], test["x0"]], axis=0),
            x1=np.concatenate([train["x1"], test["x1"]], axis=0),
            y=np.concatenate([train["y"], test["y"]], axis=0),
            bio=np.concatenate([train["bio"], test["bio"]], axis=0),
            batch=model_config["batch"],
            epoch=model_config["epoch"],
            label_dict=lab_dict,
            split=args.fold
        )
        scores = []
        i = 0
        best_score = -1.0
        for elm in cv_scores_and_models:
            s, m = elm[0], elm[1]
            scores.append(s)
            path = os.path.join(args.output, "fold-" + str(i + 1) + "-" + str(m.name) + ".h5")
            m.save(path)
            print(s, type(s))
            if s > best_score:
                best_score = float(elm[0])
                model = elm[1]
            i = i + 1
        scores = np.array(scores)
        print(f" CV Macro-F1 mean {np.mean(scores)} std {np.std(scores)}")
        df = pd.DataFrame(report)
        csv_path = os.path.join(args.output, str(args.fold) + "-fold-cv-result.csv")
        df.to_csv(csv_path)
    else:
        if not model:
            model = CEL.get_model()
        train_result = []
        history_df, model, model_store = CEL.train_model(
            model=model,
            train_data={
                "x0": train["x0"],
                "x1": train["x1"],
                "y": train["y"],
                "bio": train["bio"]
            },
            test_data={
                "x0": test["x0"],
                "x1": test["x1"],
                "y": test["y"],
                "bio": test["bio"]
            },
            label_dict=lab_dict,
            batch=model_config["batch"],
            epoch=model_config["epoch"],
            verbose=1
        )
        path = os.path.join(args.output, model.name + ".training.history.csv")
        history_df.to_csv(path)
        fig = history_df.plot(x="epoch", y=["macro_avg", "weighted_avg", "accuracy", "ser", "jer"]).get_figure()
        path = os.path.join(args.output, model.name + ".training.history.png")
        fig.savefig(path)
        with open(os.path.join(args.output, "model-hyper-param-config.json"), 'w') as jf:
            json.dump(model_config, jf)
        jf.close()
        for ik in range(len(model_store)):
            path = os.path.join(args.output, str(model.name) + "-epoch-" + str(ik) + ".h5")
            model_store[ik].save(path)

    print("=============== ALL DATA SET RESULT =====================")
    x0 = np.concatenate([train["x0"], test["x0"]], axis=0)
    x1 = np.concatenate([train["x1"], test["x1"]], axis=0)
    y = np.concatenate([train["y"], test["y"]], axis=0)
    bio = np.concatenate([train["bio"], test["bio"]], axis=0)
    f1, rdict = CEL.eval_model(
        model=model,
        eval_data={
            "x0": x0,
            "x1": x1,
            "y": y,
            "bio": bio
        },
        label_dict=lab_dict
    )
    print(f" Data Set Macro-F1 {f1}")
    for k, v in rdict.items():
        print(f" {k} >>\t {v}")

    print(f" zipping files {os.path.join(args.output, 'training-outputs.zip')}")
    with ZipFile(os.path.join(args.output, "training-outputs.zip"), 'w') as zip:
        for file in os.listdir(args.output):
            if not file.endswith(".zip"):
                zip.write(os.path.join(args.output, file))
    zip.close()
    print(f" removing files from {args.output}")
    for file in os.listdir(args.output):
        if file.endswith(".h5") or file.endswith(".csv"):
            path = os.path.join(args.output, file)
            os.remove(path)
