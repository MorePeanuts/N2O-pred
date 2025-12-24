from collections import defaultdict
from n2o_pred.data import SequentialN2ODataset

dataset = SequentialN2ODataset()

for seq_data in dataset:
    if len(set(seq_data.sowdurs)) != len(seq_data.sowdurs):
        count = defaultdict(int)
        for dur in seq_data.sowdurs:
            count[dur] += 1
        print(f'seq_id: {seq_data.seq_id}')
        for dur, num in count.items():
            if num > 1:
                print(f'sowdurs ({dur}) appears {num} times.')
