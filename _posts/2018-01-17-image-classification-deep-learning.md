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
- 본 글에서 구현한 AlexNet은, 원본 AlexNet 논문의 셋팅과 일부 다른 부분이 존재합니다. 이러한 부분을 본문 중간중간에 명시하였습니다.
- 본문에서 소개된 전체 구현 코드는 수아랩의 GitHub 저장소(*TODO: 링크 추가*)에서 자유롭게 확인하실 수 있습니다. 


## 서론

<a href="{{ site.url }}/computer-vision/2017/11/29/image-recognition-overview-1.html" target="_blank">\<이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로\></a>에서 언급한 바에 따르면, PASCAL VOC Challenge의 규정에 의거하면 이미지 인식 분야에서 총 3가지 문제를 다룬다고 하였습니다. 이번 글에서는 이들 문제 중 가장 단순한 축에 속하는 **Classification** 문제의 한 가지 사례를 가져오고, 이를 딥러닝 기술로 해결하는 과정을 여러분께 안내하고자 합니다.

본격적인 글의 전개에 앞서 중대한 사항을 하나 말씀드려야 하는데, 본 글이 *(1) 딥러닝에서의 심층 신경망 모델을 학습시키는 러닝 알고리즘에 대한 기초적인 이해가 있으며,* *(2) Python 언어 및 TensorFlow의 기초를 알고 있는* 분들을 타겟으로 한다는 점입니다. 만약 이게 갖춰지지 않은 분들이 계시다면, 아쉽지만 온/오프라인 상에 좋은 교육 자료들이 많이 있으니 이들을 먼저 공부하고 오시길 권해 드립니다. 

먼저 본 글에서 다룰 Classification 문제로는, 상대적으로 단순하면서도 많은 분들의 흥미를 끌 만한 주제인 '*개vs고양이 분류*' 문제를 다룰 것입니다. PASCAL VOC Challenge의 Classification 문제의 경우 주어진 이미지를 총 20가지 클래스(class) 중 하나로 분류하는 것이 목표였다면, '개vs고양이 분류' 문제에서는 주어진 이미지를 '개'와 '고양이'의 두 가지 클래스 중 하나로 분류하는 것이 목표라는 점에서 훨씬 단순하다고 할 수 있으며, 귀여운 개들과 고양이들의 이미지를 보는 재미가 쏠쏠하다(?)고 할 수 있습니다.

그리고 이 문제를 해결하기 위해, 2012년도 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회에서 각광을 받은 대표적인 컨볼루션 신경망(convolutional neural network)인 **AlexNet** 모델을 채택하여 학습을 진행할 것입니다. 오늘날 AlexNet보다 더 우수한 성능을 발휘한다고 알려져 있는 딥러닝 모델들이 많이 나와 있음에도 AlexNet을 쓰는 이유는, AlexNet만큼 검증이 많이 이루어진 딥러닝 모델이 드물고, 다양한 이미지 인식 문제에서 AlexNet만을 사용하고도 준수한 성능을 이끌어냈다는 사례들이 많이 보고되어 왔기 때문입니다.

### 굳이 왜?

이 쯤에서, 딥러닝을 어느 정도 알고 계시는 분들이라면 틀림없이 아래와 같은 의문을 제기하실 것 같습니다. 

> 온라인 상에 수많은 AlexNet 구현체들이 존재하는데, 굳이 이걸 여기에서 직접 제작해야 할까?

어느 정도는 일리가 있는 의문입니다. 그런데, 딥러닝 모델 구현체들을 검색하려고 조금이라고 시도해보신 분들은 아시겠지만, 온라인 상에 존재하는 구현체들을 자세히 살펴보면 하나의 파일 안에 데이터셋, 모델, 알고리즘 등과 관련된 부분들이 서로 얽히고설켜 커다란 한 덩어리로 뭉쳐있는 경우가 상당히 많습니다. 이렇게 하는 것이 맨 처음에 구현체를 신속하게 완성하고 학습 결과를 빨리 관찰하는 데 있어서는 유용한 것이 사실이나, 여기에는 몇 가지 치명적인 단점이 존재합니다.
  
우선, 이는 초심자 입장에서 딥러닝에 대해 이해하는 데 있어 효과적이지 못합니다. 보통 딥러닝의 기초 학습을 갓 마치신 분들의 머릿속에는 이런저런 개념들이 충분히 체계화되지 않은 채로 존재할 것입니다. 이 상황에서 '이제 코드 좀 짜 볼까?' 하고 TensorFlow 등의 딥러닝 프레임워크로 짜여진 예시 구현체를 찾아볼 것인데, 한 덩어리로 된 구현체들만을 계속 보다 보면 머릿속의 혼잡한 개념들이 여전히 정리가 되지 않은 형태로 남아있게 될 가능성이 높습니다.

또한, 이는 구현체 코드에 대한 유지/보수의 측면에서도 좋지 못합니다. 반복 실험 과정에서의 모델의 성능 향상을 위해, 여러분들은 기존에 마련해 놓은 구현체에 최근에 핫하다고 나온 이런저런 테크닉들을 하나둘씩 추가로 적용하여 커스터마이징(customizing)을 할 것인데, 그러다 보면 기존 코드가 걸레짝(?)처럼 되는 경우가 비일비재합니다. 그러다가 어딘지 모를 지점에서 문제가 생겨 학습이 잘 이루어지지 않는 상황이 찾아오면 문제는 더 심각해지는데, 이 얽히고설킨 코드들을 샅샅이 훑어보는 과정에서 여러분들은 극도의 스트레스에 시달리게 될 가능성이 높습니다.

지난 <a href="{{ site.url }}/machine-learning/2017/10/10/what-is-deep-learning-1.html" target="_blank">\<딥러닝이란 무엇인가?\></a>에서 *딥러닝은 머신러닝의 세부 방법론에 불과하다*고 하였으며, <a href="{{ site.url }}/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a>에서는 **머신러닝의 핵심 요소**로 *데이터셋(data set)*, *러닝 모델(learning model)*, *러닝 알고리즘(learning algorithm)* 등이 있다고 하였습니다. 여기에 학습된 러닝 모델에 대한 *성능 평가(performance evaluation)*를 추가하면, 아래와 같은 리스트가 완성됩니다.

#### 이미지 인식 문제를 위한 딥러닝 요소

- 데이터셋
- 성능 평가
- (딥)러닝 모델
- (딥)러닝 알고리즘

머신러닝의 핵심 요소를 고려하여, 딥러닝 구현체를 위와 같이 총 4가지 요소로 구분지어 이해하고자 시도한다면, 딥러닝 관련 개념들을 이해하고 이를 기반으로 자신만의 구현체를 만드는 데 매우 유용합니다. 뿐만 아니라, 새로운 딥러닝 관련 테크닉이 나오게 되더라도 그것이 위 4가지 요소 중 어느 부분에 해당하는 것인지를 먼저 파악하게 되면, 그것을 구현하는 시간을 그만큼 단축시킬 수 있습니다. 예를 들어 'DenseNet은 기존 모델에 skip connections를 무수히 많이 추가한 것이므로, 기존 러닝 모델에 이를 추가하면 되겠구나!' 내지는 'Adam은 기존의 SGD에 adaptive moment estimation을 추가한 것이므로, 기존 러닝 알고리즘에 이를 추가하면 되겠구나!' 하는 식으로 생각할 수 있을 것입니다.

그러면 지금부터, 위에서 언급한 4가지 딥러닝 요소를 기준으로 하나씩 살펴보면서, '개vs고양이 분류' 문제를 AlexNet을 사용하여 해결해보도록 하겠습니다.


## (1) 데이터셋: Asirra Dogs vs. Cats

- 데이터셋 통계량: 이미지 크기, Train/Val/Test set size, 클래스 종류 및 개수
- `datasets.asirra` 모듈: data loader, minibatch sampler
- **주의사항**: TensorFlow의 MNIST 학습용 구현체 예시에서의 feeding 방식을 그대로 채택했으나, 데이터셋 전체를 메모리에 다 로드할 수가 없다 --> 불가피하게 일부만 샘플링하여 로드
  - 메모리 효율적인 input pipeline을 제작하는 방법의 경우, 추후 별도의 글에서 다룰 예정


## (2) 성능 평가: 정확도

- 평가 척도: 정확도(accuracy)
- `learning.evaluators` 모듈


## (3) 러닝 모델: AlexNet

- `models.layers` 모듈: 
- `models.nn` 모듈: layers, loss function
- 기존 본문과의 차이점
  - 입력층의 크기로 $$224\times224\times3$$ 대신 $$227\times227\times3$$을 사용
  - 그룹 컨볼루션(grouped convolution) 대신 일반적인 형태의 컨볼루션으로 구현
  - Local response normalization 층을 제거함
  - Weight initialization 수행 시, bias 초깃값을 1.0 대신 0.1로 사용


## (4) 러닝 알고리즘: SGD+Momentum

- `learning.optimizers` 모듈: gradient descent step
- 기존 본문과의 차이점
  - Data augmentation 수행 시, PCA에 기반한 color augmentation은 수행하지 않음


## 학습 수행 및 결과

- 학습 관련 hyperparameters:
  - batch_size: 256
  - initial learning rate: 0.01
  - momentum: 0.9
  - learning rate decay rate: 0.1
  - number of epochs: 320 *FIXME*
- 정규화 관련 hyperparameters:
  - L2 weight decay: 0.0005
  - dropout probability: 0.5
- Learning curve + overfitting 여부 확인
- Test set 예시 이미지 - 예측 결과


## 결론

TODO


## References

- AlexNet 논문
  - <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" target="_blank">Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.</a>


