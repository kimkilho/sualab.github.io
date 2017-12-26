---
layout: post
title: "이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기"
date: 2018-01-17 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, computer-vision]
tags: [classification, alexnet]
comments: true
name: image-classification-deep-learning
---

지금까지 딥러닝과 이미지 인식 문제에 대해서 알아보았습니다. 해결하고자 하는 문제(이미지 인식)의 개괄을 살펴보았고 문제 해결을 위한 도구(딥러닝)에 대해 알아보았으니, 이제는 좀 더 구체적으로 이미지 인식 문제에 딥러닝을 직접 적용한 사례를 하나 제시하고, 이를 실제 구현 코드와 함께 소개해 드리고자 합니다. 지금까지의 글들이 대부분 '개념적인' 이야기들 위주였다면, 본 글에서는 코드에 기반한 '실제적인' 내용이 다뤄진다고 이해하시면 될 것 같습니다.

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - 딥러닝 알고리즘의 기본 구동 원리 및 정규화(regularization) 등의 테크닉에 대한 기초적인 내용들을 이해하고 계신 분들
  - Python 언어 및 TensorFlow의 기본적인 사용법을 알고 계신 분들
- 본 글에서는, 딥러닝 모델 및 알고리즘 구현을 위한 하나의 방식을 제시합니다. 이는 새로운 딥러닝 테크닉이 등장하였을 때, 여러분들이 사용하던 기존 모델 혹은 알고리즘에 빠르고 효과적으로 적용할 수 있도록 하기 위함이며, 그와 동시에 딥러닝 모델과 알고리즘의 작동 방식을 더 잘 이해할 수 있도록 하기 위함입니다.
- 본문에서 소개된 전체 구현 코드는 수아랩의 GitHub 저장소(*TODO: 링크 추가*)에서 자유롭게 확인하실 수 있습니다. 


## 서론

<a href="{{ site.url }}/computer-vision/2017/11/29/image-recognition-overview-1.html" target="_blank">\<이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로\></a>에서 언급한 바에 따르면, PASCAL VOC Challenge의 규정에 의거하면 이미지 인식 분야에서 총 3가지 문제를 다룬다고 하였습니다. 이번 글에서는 이들 문제 중 가장 단순한 축에 속하는 **Classification** 문제의 한 가지 사례를 가져오고, 이를 딥러닝 기술로 해결하는 과정을 여러분께 안내하고자 합니다.

본격적인 글의 전개에 앞서 중대한 사항을 하나 말씀드려야 하는데, 본 글이 *(1) 딥러닝에서의 심층 신경망 모델을 학습시키는 러닝 알고리즘에 대한 기초적인 이해가 있으며,* *(2) Python 언어 및 TensorFlow의 기초를 알고 있는* 분들을 타겟으로 한다는 점입니다. 만약 이게 갖춰지지 않은 분들이 계시다면, 아쉽지만 온/오프라인 상에 좋은 교육 자료들이 많이 있으니 이들을 먼저 공부하고 오시길 권해 드립니다. 

먼저 본 글에서 다룰 Classification 문제로는, 상대적으로 단순하면서도 많은 분들의 흥미를 끌 만한 주제인 '*개vs고양이 분류*' 문제를 다룰 것입니다. PASCAL VOC Challenge의 Classification 문제의 경우 총 20가지 클래스(class)에 해당하는 사물들을 분류하는 것이 목표였다면, '개vs고양이 분류' 문제에서는 주어진 이미지를 '개'와 '고양이'의 두 가지 클래스 중 하나로 분류하는 것이 목표라는 점에서 훨씬 단순하다고 할 수 있으며, 귀여운 개들과 고양이들의 이미지를 보는 재미가 쏠쏠하다(?)고 할 수 있습니다.

그리고 이 문제를 해결하기 위해, 2012년도 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회에서 각광을 받은 **AlexNet** 모델을 채택하여 학습을 진행할 것입니다. 오늘날 AlexNet보다 더 우수한 성능을 발휘한다고 알려져 있는 딥러닝 모델들이 많이 나와 있음에도 AlexNet을 쓰는 이유는, AlexNet만큼 검증이 많이 이루어진 딥러닝 모델이 드물고, 다양한 이미지 인식 문제에서 AlexNet만을 사용하고도 준수한 성능을 이끌어냈다는 사례들이 많이 보고되어 왔기 때문입니다.

### 굳이 왜?

이 쯤에서, 딥러닝을 어느 정도 알고 계시는 분들이라면 틀림없이 아래와 같은 의문을 제기하실 것 같습니다. 

> 온라인 상에 수많은 AlexNet 구현체들이 존재하는데, 굳이 이걸 여기에서 직접 제작해야 할까?

어느 정도 일리가 있는 의문입니다. 그런데, AlexNet 구현체들을 검색하려고 조금이라고 시도해보신 분들은 아시겠지만, 온라인 상에 존재하는 구현체들을 자세히 살펴보면 하나의 파일 안에 데이터셋, 러닝 모델, 러닝 알고리즘 등과 관련된 부분들이 서로 얽히고설켜 커다란 한 덩어리로 뭉쳐있는 경우가 상당히 많습니다. 이렇게 하는 것이 맨 처음에 구현체를 신속하게 완성하고 학습 결과를 빨리 관찰하는 데 있어서는 유용한 것이 사실이나, 여기에는 몇 가지 치명적인 단점이 존재합니다.
  
먼저, 반복 실험 과정에서의 성능 향상을 위해 여러분들이 기존에 작성해 놓은 구현체에, 최근에 핫하다고 나온 이런저런 테크닉들을 하나둘씩 추가로 적용하여 커스터마이징(customizing)을 할 것인데, 이 과정에서 코드가 걸레짝(?)처럼 되는 경우가 비일비재하다는 점을 들 수 있습니다. 그러다가 어딘지 모를 지점에서 문제가 생겨 학습이 잘 이루어지지 않는 상황이 찾아오면 문제는 더 심각해지는데, 이 얽히고설킨 코드들을 샅샅이 훑어보는 과정에서 여러분들은 극도의 스트레스에 시달리게 될 가능성이 높습니다.

또, 이렇게 한 덩어리 형태로 구현체를 작성하는 것은, 딥러닝에 대한 이해의 측면에서도 그다지 좋지 못합니다. 

*TODO*



이에 앞서, <a href="{{ site.url }}/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a>에서 살펴본 **머신러닝의 핵심 요소**를 다시 한 번 언급하고 진행하도록 하겠습니다. 

### 머신러닝의 핵심 요소

- 데이터
- 러닝 모델, 러닝 알고리즘
- 요인 추출




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

- AlexNet
  - <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" target="_blank">Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.</a>


