import argparse
from trainer import Trainer
from setting.utils import load_dataset, make_iter, Params


def train_model(config):
    params = Params('/content/drive/MyDrive/transformer/data/params.json')

    if config.mode == 'train':
        train_data, valid_data = load_dataset(config.mode)
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)

        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args(args=[])
    train_model(args)
