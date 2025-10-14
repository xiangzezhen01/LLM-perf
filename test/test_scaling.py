import pandas as pd

def test_scaling(data_path, filter_query = "2e8<N<4e9"):
    df = pd.read_csv(data_path)
    train_df = df.query(filter_query)
    test_df = df.drop(train_df.index)
    min_len = 5
    
    # scaling_fit_torch(df_trans, 5, filter_query, 7e9, True, False, prefix="Chinchilla_torch")

    # scaling_fit_fn_an_bn(df_trans, 5, f"2e8<N<{label}", True, 2-0.53, False, label, True, False)
    