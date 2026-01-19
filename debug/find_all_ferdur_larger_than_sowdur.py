from n2o_pred.data import SequentialN2ODataset

dataset = SequentialN2ODataset()

for seq_data in dataset:
    df = seq_data.numeric_dynamic
    bug_df = df[df['ferdur'] > df.index]
    if len(bug_df) > 0:
        print(bug_df)
