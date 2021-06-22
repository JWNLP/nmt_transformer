import argparse


from trainer import Trainer
from setting.utils import load_dataset, make_iter, Params


def test_model(config):
    params = Params('/content/drive/MyDrive/transformer/data/test_params.json')

    if config.mode == 'test':
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    args = parser.parse_args(args=[])
    main(args)