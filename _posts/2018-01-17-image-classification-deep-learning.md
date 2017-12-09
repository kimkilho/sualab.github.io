---
layout: post
title: "이미지 Classification 문제와 딥러닝: AlexNet 외"
date: 2018-01-17 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, computer-vision]
tags: [classification, alexnet]
comments: true
name: image-classification-deep-learning
---

## 서론

- Classification 문제에 대한 딥러닝 모델의 접근: 컨볼루션 신경망
  - Deep Convolutional Networks by Alex Krizhevsky et al. (이하 AlexNet)
- 단일 사물 분류 문제 데이터셋: Asirra Dogs vs. Cats dataset
- '딥러닝 모델 및 학습 알고리즘을, 모듈 단위로 뜯어서 이해하자'(cs231n 강좌의 모듈 구분법을 참고함)

## 데이터셋: Asirra Dogs vs. Cats dataset

- 데이터셋 통계량: 이미지 크기, Train/Val/Test set size, 클래스 종류 및 개수
- 평가 척도: 평균 정밀도(average precision) + 정밀도-재현율 곡선

## 딥러닝 모델 및 학습 알고리즘: AlexNet, SGD+Momentum

- Dataset 모듈: Input pipeline, batch loader
- CNN 모듈: layers, loss function
  - AlexNet
- Optimizer 모듈: gradient descent step
  - SGD+Momentum
- Evaluation 모듈

## 학습 수행

- 관련 hyperparameters: TODO
- Input data normalization
- Weight initialization
- Overfitting 여부 확인

## 학습 결과 평가

- 평가 척도 계산 결과
- Learning curve
- Test set 예시 이미지 - 예측 결과


## 결론

TODO


## References

TODO


