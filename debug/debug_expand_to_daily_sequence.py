import random
from n2o_pred.data import SequentialN2ODataset
from pathlib import Path


data_path = Path(__file__).parents[1] / 'datasets/data_EUR_processed.pkl'
dataset = SequentialN2ODataset(data_path)

seq_data = random.choice(dataset)
before_expand_path = Path(__file__).parent / 'output/before_expand_data.txt'
after_expand_path = Path(__file__).parent / 'output/after_expand_data.txt'
before_expand_path.parent.mkdir(parents=True, exist_ok=True)
after_expand_path.parent.mkdir(parents=True, exist_ok=True)

with before_expand_path.open('w') as f:
    seq_data.print(f)

seq_data.expand_to_daily_sequence()
with after_expand_path.open('w') as f:
    seq_data.print(f)
