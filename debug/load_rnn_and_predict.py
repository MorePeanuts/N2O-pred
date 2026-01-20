from n2o_pred.models import N2OPredictorRNN, RNNConfig
from n2o_pred.data import SequentialN2ODataset, N2ODatasetForLSTM, NUMERIC_DYNAMIC_FEATURES_RNN
from pathlib import Path


model_dir = Path(__file__).parents[1] / 'output/test_model'
model_config = RNNConfig.from_json(model_dir / 'model_config.json')
dataset = SequentialN2ODataset(numeric_dynamic_features=NUMERIC_DYNAMIC_FEATURES_RNN)
model = N2OPredictorRNN(
    num_numeric_static=dataset.get_num_numeric_static(),
    num_numeric_dynamic=dataset.get_num_numeric_dynamic(),
    categorical_static_cardinalities=dataset.get_categorical_static_cardinalities(),
    categorical_dynamic_cardinalities=dataset.get_categorical_dynamic_cardinalities(),
    model_config=model_config,
)
model.load(model_dir / 'best_model.pt')

n2o_dataset = N2ODatasetForLSTM(dataset)
model.predict(n2o_dataset)
