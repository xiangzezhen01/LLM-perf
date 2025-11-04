from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import pandas as pd
import os
pandas2ri.activate()

sk = importr('ScottKnottESD')
dir_path = 'D:\\CodePycharm\\online_dal-main\\temp\\temp_csv\\test_constrain_rebuild'
result_path = 'D:\\CodePycharm\\online_dal-main\\temp\\result\\constrain'
for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(dir_path, filename)

        df = pd.read_csv(file_path)
        if df is None:
            print("error path")
            exit()

        r_sk = sk.sk_esd(df)
        column_order = list(r_sk[3])
        ranking = pd.DataFrame(
            {
                "technique": [df.columns[i - 1] for i in column_order],
                "rank": r_sk[1].astype("int"),
            }
        )  # long format

        print(ranking)
        ranking.to_csv(os.path.join(result_path, filename), index=False)




