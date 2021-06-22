import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from setting.utils import epoch_time
from model.optim import ScheduledAdam
from model.transformer import Transformer

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        self.model = Transformer(self.params)
        self.model.to(self.params.device)

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    # 학습(training) 및 검증(validation) 진행
    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()  # 시작 시간 기록

            # 전체 학습 데이터를 확인하며
            for batch in self.train_iter:
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()
                source = batch.kor
                target = batch.eng

                # 출력 단어의 마지막 인덱스(<eos>)는 제외
                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0]

                # 입력을 할 때는 <sos>부터 시작하도록 처리
                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1])

                # 출력 단어의 인덱스 0(<sos>)은 제외
                target = target[:, 1:].contiguous().view(-1)

                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]

                # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
                loss = self.criterion(output, target)
                loss.backward()  # 기울기(gradient) 계산

                # 기울기(gradient) clipping 진행
                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)

                # 파라미터 업데이트
                self.optimizer.step()

                # 전체 손실 값 계산
                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            valid_loss = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # 모델 저장
                torch.save(self.model.state_dict(), self.params.save_model)  # "save_model": "model.pt",

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):.3f}')

    def evaluate(self):
        self.model.eval()  # 평가 모드
        epoch_loss = 0

        # 전체 평가 데이터를 확인하며
        with torch.no_grad():
            for batch in self.valid_iter:
                source = batch.kor
                target = batch.eng

                # 출력 단어의 마지막 인덱스(<eos>)는 제외
                output = self.model(source, target[:, :-1])[0]

                # 입력을 할 때는 <sos>부터 시작하도록 처리
                output = output.contiguous().view(-1, output.shape[-1])

                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]

                # 출력 단어의 인덱스 0(<sos>)은 제외
                target = target[:, 1:].contiguous().view(-1)

                # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
                loss = self.criterion(output, target)

                # 전체 손실 값 계산
                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)

    def test(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.test_iter:
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')