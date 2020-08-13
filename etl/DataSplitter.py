import pandas as pd


def make_label_wise_train_test_split(data_df: pd.DataFrame, target_per):
    c_id = data_df.cit_id.tolist()
    count = int(len(c_id) * target_per)
    train_df = data_df[:count]
    test_df = data_df[count:]
    return train_df, test_df
