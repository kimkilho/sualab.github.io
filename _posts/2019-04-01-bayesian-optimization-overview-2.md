---
layout: post
title: "Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (2)"
date: 2019-04-01 09:00:00 +0900
author: kilho_kim
categories: [Introduction, Practice]
tags: [bayesian optimization, hyperparameter optimization, gaussian process]
comments: true
name: bayesian-optimization-overview-2
---

[(이전 포스팅 보기)]({{ site.url }}{% post_url 2019-02-19-bayesian-optimization-overview-1 %})

지난 글에서 딥러닝 모델의 Hyperparamter Optimization을 위한 Bayesian Optimization 방법론의 대략적인 원리 및 행동 방식에 대한 설명을 드렸습니다. 이번 글에서는 실제 Bayesian Optimization을 위한 Python 라이브러리인 *bayesian-optimization*을 사용하여, 간단한 예시 목적 함수의 최적해를 탐색하는 과정을 먼저 소개하고, 실제 딥러닝 모델의 최적 hyperparameter를 탐색하는 과정을 안내해 드리도록 하겠습니다.

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - 딥러닝 알고리즘의 기본 구동 원리 및 정규화(regularization) 등의 테크닉에 대한 기초적인 내용들을 이해하고 계신 분들
  - Python 언어 및 TensorFlow의 기본적인 사용법을 알고 계신 분들
- 본격적인 시작에 앞서, 여러분의 Python 환경 상에 <a href="https://pypi.org/project/bayesian-optimization/" target="_blank">bayesian-optimization</a> 라이브러리를 먼저 설치해 주시길 바랍니다. 이는 PyPI에서 *bayesian-optimization*이라는 이름의 패키지로 제공되며, pip로 설치하실 수 있습니다.
- 본 글의 중반부에 소개된, 예시 목적 함수에 대한 최적해를 탐색하는 과정은 <a href="https://github.com/sualab/sualab.github.io/blob/master/assets/notebooks/{{ page.name }}.ipynb" target="_blank">여기</a>에서 확인하실 수 있습니다.
- 본 글의 후반부에서는 지난 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">\<이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기\></a> 글에서 사용했던 AlexNet 구현체를 그대로 가져와서, 딥러닝 모델 학습과 관련된 최적의 hyperparameter를 탐색하는 과정에 대해서만 *bayesian-optimization* 라이브러리를 사용하는 방법을 중심으로 설명합니다. 본문을 따라 구현체를 작성하고 시험적으로 구동해 보고자 하시는 분들은, 아래 사항들을 참조해 주십시오.
	- 만일 \<이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기\> 글을 읽어보지 않으셨다면, 먼저 해당 글을 읽어보시면서 AlexNet 구현체 및 개vs고양이 분류 데이터셋에 대한 준비를 미리 완료해 주시길 바랍니다.
  - 본 글에서 최적 hyperparameter 탐색을 수행하는 전체 구현체 코드는 <a href="https://github.com/sualab/asirra-dogs-cats-classification" target="_blank">수아랩의 GitHub 저장소</a>에서 자유롭게 확인하실 수 있습니다.
    - 전체 구현체 코드 원본에는 모든 주석이 (일반적인 관습에 맞춰) 영문으로 작성되어 있으나, 본 글에서는 원활한 설명을 위해 이들을 한국어로 번역하였습니다.


## bayesian-optimization 라이브러리 소개

오늘날 Bayesian Optimization을 다양한 문제에 원활하게 적용할 수 있도록 하는 여러 Python 라이브러리가 공개되어 있습니다. 이들 중에서 Fernando Nogueira 씨가 제작한 <a href="https://github.com/fmfn/BayesianOptimization" target="_blank"><i>bayesian-optimization</i></a> 라이브러리를 간략하게 소개해 드리고, 간단한 예시 목적 함수의 최적해를 탐색하는 방법에 대해 안내해 드린 후, 실제 딥러닝 모델의 hyperparameter 탐색을 위해서 어떻게 사용할 수 있는지를 집중적으로 설명하도록 하겠습니다.

{% include image.html name=page.name file="bayesian-optimization-library.png" description="bayesian-optimization 라이브러리" class="full-image" %}

우선 *bayesian-optimization* 라이브러리는 Surrogate Model로 Gaussian Process(이하 GP)를 채택하는데, 이와 관련된 세부 계산 과정들을 한두 개의 단순한 함수들로 wrapping해 놓았기 때문에, GP의 계산과 관련된 자세한 과정을 모르고 있더라도 아주 쉽고 간결하게 사용할 수 있다는 게 최대 장점입니다. 지난 글에서 보여 드렸던, Bayesian Optimization의 필수 요소인 Surrogate Model과 Acquisition Function의 행동적 특징만 제대로 이해하고 있으면, 해당 라이브러리를 사용하는 데 아무런 문제가 없다고 할 수 있습니다.

또한 *bayesian-optimization* 라이브러리는 다른 Bayesian Optimization 라이브러리에 비해 dependency가 적기 때문에, 본격적인 사용에 앞서 추가로 설치해야 하는 라이브러리가 *numpy*, *scipy*, *scikit-learn* 등에 불과하다는 점도 장점이라고 할 수 있습니다. 여기에 덧붙여 다양한 사용 시나리오에 대한 문서화 및 시각화 등이 상세하게 잘 되어 있기 때문에, 라이브러리의 사용법을 빠르게 터득할 수 있다는 점 또한 장점으로 꼽을 만합니다.

다른 Bayesian Optimization 라이브러리와 비교하여 *bayesian-optimization* 라이브러리의 단점이 있다면 Surrogate Model로 GP 외의 것은 지원하지 않는다는 점일 것입니다. 다만 지난 글에서 언급했던 학습률(learning rate), L2 정규화 계수(L2 regularization coefficient) 등과 같이 연속형(continuous) hyperparameter를 탐색하는 데에 한해서는, GP만으로도 준수한 결과를 기대해 볼 수 있습니다.

(그러나 아키텍처(architecture) 또는 데이터 증강(data augmentation) 등과 같은 요소들과 연관되어 있는 이산형(discrete) hyperparameter를 탐색고자 할 경우에는 단순 GP만으로는 부족한 측면이 있습니다. 만일 탐색 대상 hyperparameter가 순서형(ordinal)인 경우에는 이를 연속형으로 간주하여 GP를 통해 탐색할 수는 있습니다만, 이것이 완벽한 해법은 아님을 감안해야 합니다.)


## 간단한 예시 목적 함수의 최적해 탐색

*bayesian-optimization* 라이브러리의 기본적인 사용 방법을 파악하기 위해, 입력값이 $$x$$ 하나인 간단한 예시 목적 함수 $$f(x)$$의 최적해 $$x^*$$를 Bayesian Optimization으로 탐색하는 방법을 먼저 소개해 드리도록 하겠습니다. 통상적으로 Bayesian Optimization의 공략 대상이 되는 함수는 그 표현식을 명시적으로 알지 못하는 black-box function인 경우가 보통이라고 하였으나, 이 경우에는 이해를 돕기 위한 목적으로 표현식을 알고 있는 함수를 대신 사용한다고 봐 주시면 됩니다.

참고로, 본 과정은 *bayesian-optimization* 라이브러리의 GitHub 저장소에서 제공되는 <a href="https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb" target="_blank">step-by-step visualization 문서</a>의 내용을 상당 부분 참조하여 재구성한 것입니다. 실제 Bayesian Optimization 과정에 대한 시각화 방법이 궁금하시다면, 해당 페이지를 참조해 주시길 바랍니다.

$$
f(x) = e^{-(x-3)^2} + e^{-(3x-2)^2} + \frac{1}{x^2+1}
$$

여기에서는 위와 같은 예시 목적 함수의 *최댓값*을 탐색하는 상황을 가정하겠습니다. 상기 표현식을 알고 있다는 가정 하에서 실제 최적화 알고리즘을 통해 계산해 보면, $$x=0.631$$ 부근에서 최댓값 $$f(0.631)=1.707$$이 발생합니다.

#### 0. import, import, import...

맨 먼저 *bayesian-optimization* 라이브러리의 핵심 요소라고 할 수 있는 `BayesianOptimization` 클래스를 import하고, *numpy* 패키지를 추가로 import합니다.

```python
from bayes_opt import BayesianOptimization
import numpy as np
```

#### 1. 입력값 및 목적 함수 정의

다음으로 입력값 `x`를 인자로 하는 목적 함수 `target`을 아래와 같이 정의합니다. 해당 목적 함수를 따로 플롯팅해 보면 그 아래 그림과 같이 나타납니다.

```python
def target(x):
    return np.exp(-(x-3)**2) + np.exp(-(3*x-2)**2) + 1/(x**2+1)
```

{% include image.html name=page.name file="target-function-plot.png" description="예시 목적 함수의 플롯팅 결과" class="full-image" %}

#### 2. BayesianOptimization 객체 생성

다음으로 `BayesianOptimization` 객체를 하나 생성합니다. 코드 상에는 드러나 있지 않으나, 이 `BayesianOptimization` 객체에는 Surrogate Model인 GP가 기본적으로 내장되어 있으며, 이는 실제로는 해당 객체 내부의 멤버 변수 `_gp`로 표현됩니다.

```python
bayes_optimizer = BayesianOptimization(target, {'x': (-2, 6)}, 
                                       random_state=0)
```

`BayesianOptimization` 객체를 생성할 시, 앞쪽 두 개의 인자는 필수적으로 입력해 줘야 하는 것들에 해당합니다. 첫 번째 인자(`target`)는 최적해를 탐색하고자 하는 목적 함수 $$f(x)$$를, 두 번째 인자(`{'x': (-2, 6)}`)는 입력값 $$x$$의 탐색 대상 구간 $$(a, b)$$를 dictionary 형태로 받습니다. 입력값 $$x$$의 탐색 대상 구간은, 미지의 목적 함수에 대하여 현재까지 여러분들이 인지하고 있는 사전 지식에 기반하여 적절히 설정하되, 추후 필요한 경우 범위를 좁히거나 넓혀 재설정할 수도 있습니다.

한편 `random_state` 인자는 Bayesian Optimization 상의 랜덤성이 존재하는 부분(e.g. 다음 입력값 후보 추출 등)을 통제할 수 있도록 random seed를 입력해 주기 위한 목적으로 입력됩니다. 랜덤성을  통제할 필요가 없을 경우 입력하지 않아도 무방합니다.

#### 3. Bayesian Optimization 실행

마지막으로, 생성한 `BayesianOptimization` 객체의 `maximize` 함수를 호출하여, 내부 멤버 변수 `_gp`를 반복적으로 업데이트하면서 실제 Bayesian Optimization 과정을 수행합니다.

```python
bayes_optimizer.maximize(init_points=2, n_iter=14, acq='ei', xi=0.01)
```

`maximize` 함수의 `init_points` 인자는, 맨 처음에 일부 Random Search 방법으로 조사할 입력값-함숫값 점들의 갯수($$n$$)을 나타냅니다. 앞서 설정한 구간 내에서 해당 `init_points` 개의 입력값들을 랜덤하게 샘플링한 뒤, 앞서 정의한 `target` 함수에 대입하여 이들에 대한 함숫값들을 각각 계산하여 저장합니다. 이 때, `init_points`를 큰 값으로 설정할수록, 전체 구간에 대한 충분한 사전 탐색을 수행해 놓을 수 있으나, 그만큼 시간이 많이 소요됩니다.

다음으로 `n_iter` 인자는, 처음 $$n$$개의 입력값-함숫값 점들을 조사한 후, 조사된 입력값-함숫값 점들의 총 갯수가 $$N$$개에 도달할 때까지, Bayesian Optimization 방법을 통해 추가로 조사할 입력값-함숫값 점들의 총 갯수($$N-n$$)를 나타냅니다. 예를 들어, 총 16개의 입력값-함숫값 점들을 확보하여 최적값을 탐색하고자 할 시, `init_points=2`로 설정하여 처음 2개의 입력값-함숫값 점들을 조사하고자 한다면, `n_iter=14`로 설정하면 나머지 14개의 입력값-함숫값 점들은 Bayesian Optimization 방법으로 조사하게 됩니다. 

이어지는 `acq` 인자는, 현재 *bayesian-optimization* 라이브러리에서 제공하는 Acquisition Function들 중 어느 것을 사용할지를 명시하는 부분입니다. 만일 Expected Improvement(EI)를 Acquisition Function으로 사용하고자 한다면, `acq='ei'`로 설정하면 됩니다. 이에 결부되어 나오는 마지막 인자가 `xi`인데, 이는 지난 글에서 수식으로 보여드린 바 있는, exploration-explotation 간의 상대적 강도를 조절해 주는 파라미터 $$\xi$$입니다. *bayesian-optimization*에서 `xi`의 기본값은 0.0이나, 그보다는 exploration의 강도를 좀 더 높이기 위해 0.01 정도로 설정해 주는 것이 무난합니다.

#### 4. Bayesian Optimization 최종 결과

처음 랜덤한 방식의 2회, 이후 14회의 반복 회차에 걸쳐 Bayesian Optimization을 수행한 결과는 아래와 같이 출력됩니다. 

{% include image.html name=page.name file="bayesian-optimization-result-example.png" description="" class="full-image" %}

'iter'는 반복 회차, 'target'은 목적 함수의 값, 'x'는 입력값을 나타냅니다. 현재 회차 이전까지 조사된 함숫값들과 비교하여, 현재 회차에 최댓값이 얻어진 경우, *bayesian-optimization* 라이브러리는 이를 자동으로 다른 색 글자로 표시하는 것을 확인할 수 있습니다. 최종적으로 업데이트가 완료된 `BayesianOptimization` 객체 내부의 멤버 변수 `_gp`를 실제 목적 함수와 함께 도시하면 아래와 같습니다.

{% include image.html name=page.name file="estimated-function-plot.png" description="예시 목적 함수에 대한 추정의 플롯팅 결과<br><small>(파란색 실선: 목적 함수(Objective Function), 검은색 점선: GP를 통해 추정한 평균 함수(Prediction),<br>붉은색 점: 조사된 입력값-함숫값 점(Observations), 하늘색 영역: GP를 통해 추정한 표준편차(불확실성; Uncertain area))</small>" class="full-image" %}

총 16회의 입력값-함숫값 조사를 거쳐, GP를 통해 추정한 평균 함수 $$\mu(x)$$의 결과가 실제 목적 함수 $$f(x)$$와 거의 일치하는 것을 확인할 수 있습니다. 추정된 평균 함수에 대하여 최적화 알고리즘을 적용해 보면, $$x=0.629$$ 부근에서 최댓값 $$\mu(0.629)=1.707$$이 발생합니다. 앞서 계산했던 실제 최적 입력값인 $$x=0.631$$과 약 $$0.002$$ 정도의 차이만 존재하는 것을 확인할 수 있습니다.


## Bayesian Optimization을 사용한 딥러닝 모델의 주요 hyperparameter 탐색

*bayesian-optimization* 라이브러리를 사용하면, 위와 같이 불과 몇 줄의 코드만으로도 목적 함수의 최적해에 대한 탐색이 가능합니다. 위에서 소개한 과정을 그대로 따라하되, '**특정 hyperparameter를 입력값으로 받아, 이를 딥러닝 모델 학습에 적용하였을 시의 검증 데이터셋에 대한 성능 결과 수치를 출력값으로 제시하는 함수**'를 목적 함수로 설정하면, *bayesian-optimization* 라이브러리를 그대로 딥러닝 모델의 Hyperparameter Optimization에 적용할 수 있습니다.

### '개vs고양이 분류' 문제 해결을 위한 AlexNet 구현체

실제 딥러닝 모델 학습 예시로, 지난 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">\<이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기\></a> 글에서 사용했던 AlexNet 구현체를 그대로 사용, Asirra Dogs vs. Cats 데이터셋으로 해당 모델을 학습하고 이를 '개vs고양이 분류' 문제에 적용하는 과정을 다시 한 번 채택하기로 하였습니다. AlexNet으로 개vs고양이를 분류하는 과정에 대한 자세한 설명은, 해당 글을 다시 참조해 주시길 바랍니다.

지난 글에서 개vs고양이 분류를 위해 AlexNet을 학습할 당시에는, 사전에 지정한 하나의 hyperparameter들의 조합만을 사용하여 모델을 1회만 학습한 바 있습니다. 당시 설정했던 hyperparameter들 중 모델의 학습 성패 및 일반화 성능에 지대한 영향을 미칠 만한 것은 (decay 적용 전의) **초기 학습률(initial learning rate)**과 **L2 정규화 계수(L2 weight decay)**라고 할 수 있습니다. 

- 초기 학습률(initial learning rate): 0.01
- L2 정규화 계수(L2 weight decay): 0.0005

지난 글에서는 초기 학습률 및 L2 정규화 계수를 위와 같이 설정한 바 있습니다. 과연 위의 설정값들이 AlexNet의 일반화 성능을 극대화시키기 위한 '최적'의 hyperparameter 값들에 해당하는지 확인해 보기 위해, Bayesian Optimization을 통해 초기 학습률 및 L2 정규화 계수에 대한 탐색을 수행해 보도록 하겠습니다.

### train-with-bo.py 스크립트

기존 AlexNet 구현체에서 실제 학습을 수행하는 과정을 담은 것이 `train.py` 스크립트였는데, 여기에 *bayesian-optimization* 라이브러리를 적용한 Bayesian Optimization 과정을 추가하여 새로운 `train-with-bo.py` 스크립트를 구현하였습니다.

```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# 중간 점검
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. 학습 수행 및 성능 평가를 위한 기본 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
np.save('/tmp/asirra_mean.npy', image_mean)    # 평균 이미지를 저장
hp_d['image_mean'] = image_mean

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 200

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: 정규화 관련 하이퍼파라미터
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: 성능 평가 관련 하이퍼파라미터
hp_d['score_threshold'] = 1e-4


""" 3. 특정한 초기 학습률 및 L2 정규화 계수 하에서 학습을 수행한 후, 검증 성능을 출력하는 목적 함수 정의 """
def train_and_validate(init_learning_rate_log, weight_decay_log):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    hp_d['init_learning_rate'] = 10**init_learning_rate_log
    hp_d['weight_decay'] = 10**weight_decay_log

    model = ConvNet([227, 227, 3], 2, **hp_d)
    evaluator = Evaluator()
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

    sess = tf.Session(graph=graph, config=config)
    train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)

    # 검증 정확도의 최댓값을 목적 함수의 출력값으로 반환
    best_val_score = np.max(train_results['eval_scores'])

    return best_val_score


""" 4. BayesianOptimization 객체 생성, 실행 및 최종 결과 출력 """
bayes_optimizer = BayesianOptimization(
    f=train_and_validate,
    pbounds={
        'init_learning_rate_log': (-5, -1),    # FIXME
        'weight_decay_log': (-5, -1)            # FIXME
    },
    random_state=0,
    verbose=2
)

bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)    # FIXME

for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)
```

`train-with-bo.py` 스크립트에서는 다음의 4단계 과정을 거칩니다.

1. 원본 학습 데이터셋을 메모리에 로드하고, 이를 학습 데이터셋(80%)과 검증 데이터셋(20%)으로 나눈 뒤 각각을 사용하여 `DataSet` 객체를 생성함
2. 학습 수행 및 성능 평가를 위한 (조사 대상 외의) 기본 hyperparameter들을 설정함
3. 특정한 초기 학습률 및 L2 정규화 계수 하에서 학습을 수행한 후, 검증 성능을 출력하는 목적 함수 `train_and_validate`을 정의함
  - 함수 내부에서 `ConvNet` 객체, `Evaluator` 객체 및 `Optimizer` 객체를 생성하고, TensorFlow Graph와 Session을 초기화한 뒤, `Optimizer.train` 함수를 호출하여 모델 학습을 수행함
  - 학습 진행 과정에서, 검증 데이터셋에 대하여 매 epoch마다 얻어진 모델의 예측 정확도(검증 정확도) 중 최댓값을 `train_and_validate` 함수가 반환하도록 함
4. 앞서 정의한 목적 함수 및 입력값들의 탐색 대상 구간을 인자로 입력하여 `BayesianOptimization` 객체를 생성한 후, 초기 Random Search 및 이후 Bayesian Optimization을 통해 조사할 입력값-함숫값 점들의 갯수들을 각각 인자로 입력하여 `maximize` 함수를 호출하고, 모든 과정이 끝나면 최종 결과를 출력함

이 때, `train_and_evaluate` 함수의 인자로 받는 `init_learning_rate_log`와 `weight_decay_log`는, 각각 **초기 학습률과 L2 정규화 계수를 <a href="https://en.wikipedia.org/wiki/Logarithmic_scale" target="_blank">base-10 log scale</a>로 표현한 것**임을 유의하셔야 합니다. 예를 들어 `init_learning_rate_log`의 값이 $$\alpha$$인 경우 실제 학습 시의 초기 학습률은 $$10^{\alpha}$$로 설정되며, `weight_decay_log`의 값이 $$\lambda$$인 경우 실제 학습 시의 L2 정규화 계수는 $$10^{\lambda}$$로 설정됩니다.

그리고, `BayesianOptimization` 객체 생성 및 `maximize` 함수 호출 시 인자들과 관련하여, `FIXME`로 표시된 부분은 여러분의 상황과 기호에 맞춰 수정하실 수 있습니다. 본 글에서 Bayesian Optimization을 수행할 당시의 해당 인자 값들을 아래와 같이 설정하였습니다.

- 입력값들의 탐색 대상 구간 $$(a, b)$$ (`pbounds`)
  - 초기 학습률(initial learning rate; base-10 log scale): (-5, -1)
  - L2 정규화 계수(L2 weight decay; base-10 log scale): (-5, -1)
- 처음 랜덤하게 조사할 입력값-함숫값 점들의 갯수 $$n$$ (`init_points`): 3
- Bayesian Optimization 방법을 통해 추가로 조사할 입력값-함숫값 점들의 갯수 $$N-n$$ (`n_iter`): 27 

### Bayesian Optimization을 통한 hyperparameter 탐색 결과

#### 초기 학습률 및 L2 정규화 계수에 대한 Bayesian Optimization 수행 과정

`train-with-bo.py` 스크립트를 실행하여, 최적의 초기 학습률 및 L2 정규화 계수를 탐색하는 과정을 아래 그림과 같이 시각화하였습니다. 입력값이 두 가지이므로 이를 2차원 평면 상의 히트맵(heatmap) 형태로 나타냈으며, 빨간색에 가까울수록 그 값이 크고, 파란색에 가까울수록 그 값이 작음을 의미합니다. 실제 조사된 입력값-함숫값 점의 경우, 검은색 점으로 표시했습니다.

{% include image.html name=page.name file="2d-bayesian-optimization-process.gif" description="(초기 학습률, L2 정규화 계수)에 대한 2차원 Bayesian Optimization(GP, EI) 수행 과정<br><small>(각각 구간 $$(-5, -1)$$, $$(-5, -1)$$에서 최초 3개($$n=3$$), 총 30개($$N=30$$)의 점에 대한 반복 조사 결과<br>좌측: GP를 통해 추정한 평균 함수 $$\mu(x,y)$$, 중앙: GP를 통해 추정한 표준편차 $$\sigma(x,y)$$,<br>우측: GP의 확률적 추정 결과에 대한 EI 함수 계산 결과(십자 표시: 다음 조사 대상 점의 위치); random_seed=0)</small>" class="full-image" %}

위 과정을 거쳐 초기 학습률과 L2 정규화 계수에 대한 탐색을 수행한 결과, 최적의 값들이 아래와 같이 확인되었습니다.

- 최적 초기 학습률(initial learning rate): $$10^{-1.7927} \approx 0.016118$$
- 최적 L2 정규화 계수(L2 weight decay): $$10^{-4.5997} \approx 0.000025$$

#### 최적의 hyperparameter 값들을 채택한 학습 곡선

위와 같이 찾은 최적 hyperparameter 값들을 그대로 채택, 기존 `train.py` 스크립트를 실행하여 실제 학습을 수행한 결과, 아래의 정보들을 담은 학습 곡선(learning curve)을 얻을 수 있었습니다.

- 매 반복 회차에서의 손실 함수의 값
- 매 epoch에 대하여 (1) 학습 데이터셋으로부터 추출한 미니배치(minibatch)에 대한 모델의 예측 정확도(학습 정확도)와 (2) 검증 데이터셋에 대한 모델의 예측 정확도(검증 정확도)

{% include image.html name=page.name file="optimal-learning-curve-result.svg" description="학습 곡선 플롯팅 결과<br><small>(파란색: 학습 데이터셋 정확도, 빨간색: 검증 데이터셋 정확도)</small>" class="large-image" %}

학습 과정 말미에서, 검증 정확도가 0.9384일 때의 모델 파라미터들을 최종적으로 채택하여, 테스트를 위해 저장하였습니다.

#### 테스트 결과

테스트 결과 측정된 정확도는 **0.93688**로 확인되었습니다. 이전 글에서 측정된 정확도가 **0.92768**이었는데, 이를 약 1% 가량 상회하는 결과가 얻어졌습니다. 지금까지의 과정을 다시금 곱씹어 보면, 제한된 양의 학습 데이터셋만을 활용해서, 두 가지 hyperparameter에 대하여 단 30회의 자동화된 반복적 조사를 통해 최적의 hyperparameter 값을 찾아낸 결과라고 할 수 있습니다. 

매 회 조사 결과를 사람이 직접 관찰한 뒤 그 다음 번 조사 대상을 설정하여 진행하는 Manual Search나, 매 회 새로운 조사 수행 시 '사전 지식'이 반영되지 않아 다소 불필요한 조사를 반복하게 되는 Grid Search 및 Random Search에 비해, 매 회 조사 대상 선정을 자동화하였으면서 동시에 확률적 추정을 통해 '사전 지식'을 충분히 반영하였다는 측면에서 이들보다 효율적으로 얻어진 결과라고 생각됩니다. 


## 결론

지난 글에서 딥러닝 모델의 Hyperparamter Optimization을 위한 Bayesian Optimization 방법론의 대략적인 원리 및 행동 방식에 대하여 알아보았고, 본 글에서는 이를 구현하기 위한 Python 라이브러리인 *bayesian-optimization*에 대한 소개 및 기본적인 사용법을 안내해 드렸습니다. Surrogate Model로 GP, Acquision Function으로 EI를 채택하여, 간단한 예시 목적 함수의 최적해를 탐색하는 과정을 진행하였으며, 그 다음 실제 AlexNet 모델의 초기 학습률 및 L2 정규화 계수의 최적값을 탐색하는 과정을 진행하였습니다.

더욱 직접적인 비교를 위해, 여러분께서 초기 학습률 및 L2 정규화 계수에 대한 Grid Search와 Random Search 또한 직접 구현하고 실행하시어, 이들을 통해 최적의 hyperparameter를 탐색하는 과정을 Bayesian Optimization 과정과 직접 비교해 보시길 권장드립니다. 보다 드라마틱한 차이를 확인하실 수 있을 것이라고 생각합니다.


## References

- Fernando Nogueira, bayesian-optimization: A Python implementation of global optimization with gaussian processes.
  - <a href="https://github.com/fmfn/BayesianOptimization" target="_blank">https://github.com/fmfn/BayesianOptimization</a>
- Logarithmic scale
  - <a href="https://en.wikipedia.org/wiki/Logarithmic_scale" target="_blank">Wikipedia contributors. "Logarithmic scale." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 13 Mar. 2019. Web. 1 Apr. 2019.</a>
