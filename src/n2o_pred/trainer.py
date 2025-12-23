from loguru import logger
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from .data import SequentialN2ODataset
from .utils import set_global_seed


def simplest_training(
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
