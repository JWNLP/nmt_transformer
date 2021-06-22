import os
import pickle
import argparse  # 코랩에선 config = parser.parse_args(args=[])
#import easydict  # 코랩에선 argparse 대신 easydict

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    """
    print(f'Now building soy-nlp tokenizer . . .')

    data_dir = '/content/drive/MyDrive/transformer/data/'
    train_file = os.path.join(data_dir, 'corpus.csv')

    df = pd.read_csv(train_file, encoding='utf-8')

    # if encounters non-text row, we should skip it
    kor_lines = [row.korean
                 for _, row in df.iterrows() if type(row.korean) == str]

    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}

    with open('/content/drive/MyDrive/transformer/pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    pickle_tokenizer = open('/content/drive/MyDrive/transformer/pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # include lengths of the source sentences to use pack pad sequence
    kor = ttd.Field(tokenize=tokenizer.tokenize,
                    lower=True,
                    batch_first=True)

    eng = ttd.Field(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = '/content/drive/MyDrive/transformer/data/'
    train_file = os.path.join(data_dir, 'train.csv')

    # train_data = pd.read_csv(train_file, encoding='utf-8') 이 코드에서 ParserError가 발생하면 engine이랑 delimiter 상세히 기입하기
    # ParserError: Error tokenizing data. C error: EOF inside string starting at row 27202
    train_data = pd.read_csv(train_file, encoding='utf-8', engine='python', delimiter=',')
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')

    kor.build_vocab(train_data, max_size=config.kor_vocab)
    eng.build_vocab(train_data, max_size=config.eng_vocab)

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    with open('/content/drive/MyDrive/transformer/pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('/content/drive/MyDrive/transformer/pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    config = parser.parse_args(args=[])
    # Jupyter notebook에서 argparse 이용할 때, 굳이 easydict으로 안바꾸고
    # 마지막 config = parser.parse_args() 괄호안에 args=[] 추가

    # parser = easydict.EasyDict({
    #    "kor_vocab":55000,
    #    "eng_vocab":30000
    # })

    # config = parser.parse_args(config=[])

    build_tokenizer()
    build_vocab(config)