from loguru import logger
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from .evaluation import compute_regression_metrics
from .models import N2OPredictorRF, RandomForestConfig
from .data import SequentialN2ODataset, LABELS
from .utils import set_global_seed


class SimplestTrainer:
    def __init__(self):
        pass

    def simplest_training(
        self,
        model_type='rf',
        random_seed=42,
        train_split=0.9,
        data_path=None,
        output_path=None,
    ):
        logger.info(f'Random seed: {random_seed}')
        set_global_seed(random_seed)

        if data_path is None:
            data_path = Path(__file__).parents[2] / 'datasets/data_EUR_processed.pkl'
        else:
            data_path = Path(data_path)
        dataset = SequentialN2ODataset(data_path)
        logger.info(f'Load sequential dataset from {data_path}, total sequences: {len(dataset)}')

        if output_path is None:
            output_path = (
                Path(__file__).parents[2]
                / f'output/simplest_training_{datetime.now().strftime("%m%d_%H%M%S")}'
            )
        else:
            output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        indices = list(range(len(dataset)))
        test_ratio = (1.0 - train_split) / 2
        train_val_indices, test_indices = train_test_split(
            indices, train_size=1.0 - test_ratio, random_state=random_seed
        )
        val_ratio = test_ratio / (1.0 - test_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices, train_size=1.0 - val_ratio, random_state=random_seed
        )
        logger.info(
            f'Dataset split: {len(train_indices)} for training, {len(val_indices)} for validation, {len(test_indices)} for test.'
        )

        # 创建训练集、测试集、验证集
        train_dataset = dataset[train_indices]
        val_dataset = dataset[val_indices]
        test_dataset = dataset[test_indices]

        # TODO: train specilized model based on model type
        match model_type:
            case 'rf':
                self.train_random_forest(train_dataset, val_dataset, test_dataset, output_path)
            case 'lstm':
                self.train_lstm_model(train_dataset, val_dataset, test_dataset, output_path)

    def train_random_forest(
        self,
        train_dataset: SequentialN2ODataset,
        val_dataset: SequentialN2ODataset,
        test_dataset: SequentialN2ODataset,
        output_path: Path,
    ):
        logger.info('Expand sequence data into tabular data...')
        train_df = train_dataset.flatten_to_dataframe()
        val_df = val_dataset.flatten_to_dataframe()
        test_df = test_dataset.flatten_to_dataframe()

        # 初始化随机森林模型，并在训练集上训练
        # TODO: RF模型训练应该更改为交叉验证
        config = RandomForestConfig()
        logger.info(f'Initialize the model with the following parameters: {config}')
        model = N2OPredictorRF(**config.to_dict())
        model.fit(train_df)
        logger.info(f'Model training completed, model complexity: {model.count_parameters()}')

        # 预测
        train_preds = model.predict(train_df)
        val_preds = model.predict(val_df)
        test_preds = model.predict(test_df)

        target_col = LABELS[0]
        train_targets = train_df[target_col].values
        val_targets = val_df[target_col].values
        test_targets = test_df[target_col].values

        # 计算评测指标
        train_metrics = compute_regression_metrics(train_targets, train_preds)
        val_metrics = compute_regression_metrics(val_targets, val_preds)
        test_metrics = compute_regression_metrics(test_targets, test_preds)
        logger.info(
            f'Training set - R2: {train_metrics["R2"]:.4f}, RMSE: {train_metrics["RMSE"]:.4f}'
        )
        logger.info(
            f'Validation set - R2: {val_metrics["R2"]:.4f}, RMSE: {val_metrics["RMSE"]:.4f}'
        )
        logger.info(f'Test set - R2: {test_metrics["R2"]:.4f}, RMSE: {test_metrics["RMSE"]:.4f}')

        # 保存模型、预测结果、评测结果
        model_path = output_path / 'random_forest_n2o_predictor.pkl'
        model.save(model_path)
        logger.info(f'Random forest N2O predictor has been saved to {model_path}')
        # TODO: 预测结果和评测结果保存

        # 获取特征重要性
        feature_importances = model.get_feature_importances()
        logger.info(f'Feature importances: {feature_importances}')

    def train_lstm_model(
        self,
        train_dataset: SequentialN2ODataset,
        val_dataset: SequentialN2ODataset,
        test_dataset: SequentialN2ODataset,
        output_path: Path,
    ):
        pass
