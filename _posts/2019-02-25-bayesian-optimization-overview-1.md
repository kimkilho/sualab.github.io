---
layout: post
title: "Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (1)"
date: 2019-02-15 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, computer-vision]
tags: [bayesian optimization, hyperparameter optimization, gaussian process]
comments: true
name: bayesian-optimization-overview-1
---

지난 <a href="{{ site.url }}/machine-learning/computer-vision/2018/09/28/nasnet-review.html" target="_blank">\<Learning Transferable Architectures for Scalable Image Recognition 리뷰\></a> 글의 서두에서 'AutoML'이라는 주제에 대해 간단히 소개해 드린 적이 있습니다. AutoML을 한 문장으로 표현하자면 'Machine Learning으로 설계하는 Machine Learning'이라고 하였으며, 현재 3가지 방향으로 연구가 진행된다고 하였습니다.

1. Automated Feature Learning
2. Architecture Search
3. Hyperparameter Optimization

본 글에서는, 위 연구 방향들 중 3번째 항목인 'Hyperparameter Optimization'에 대해 소개해 드리고, 딥러닝 분야에서의 Hyperparameter Optimization을 위한 주요 방법론들에 대한 대략적인 설명과 더불어, '학습'의 관점에서 최적의 hyperparameter를 탐색하기 위한 방법 중 하나인 'Bayesian Optimization'에 대하여 안내해 드리고자 합니다. 그리고, Bayesian Optimization을 위한 Python 라이브러리 중 하나인 <a href="https://github.com/fmfn/BayesianOptimization" target="_blank">bayesian-optimization</a>을 소개해 드리고, 실제로 이를 사용하여 이미지 Classification을 위한 딥러닝 모델의 주요 hyperparameter들의 최적값을 찾는 과정을 안내해 드리고자 합니다. 

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - 딥러닝 알고리즘의 기본 구동 원리 및 정규화(regularization) 등의 테크닉에 대한 기초적인 내용들을 이해하고 계신 분들
  - Python 언어 및 TensorFlow의 기본적인 사용법을 알고 계신 분들
- 본 글의 목적은, 독자 여러분들로 하여금 Bayesian Optimization에 대한 깊은 이해를 유도하는 것이 아니라, **Bayesian Optimization에 대한 기초적인 이해만을 가지고 이를 딥러닝 모델의 Hyperparameter Optimization에 원활하게 적용할 수 있도록 하는 것**입니다. 이에 따라 본 글에서는 Bayesian Optimization의 대략적인 원리를 설명하는 단계에서 딥러닝 외적인 수학적 내용들에 대한 언급은 가급적 피하고자 하였으나, 최소한의 설명을 위해 수학적 내용이 약간은 등장할 수 있음을 양지해 주시길 바랍니다.
- 본문에서는 지난 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">\<이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기\></a> 글에서 사용했던 AlexNet 구현체를 그대로 가져와서, 딥러닝 모델 학습과 관련된 최적의 hyperparameter를 탐색하는 과정에 대해서만 bayesian-optimization 라이브러리를 사용하는 방법을 중심으로 설명합니다. 본문을 따라 구현체를 작성하고 시험적으로 구동해 보고자 하시는 분들은, 아래 사항들을 참조해 주십시오.
	- 만일 \<이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기\> 글을 읽어보지 않으셨다면, 먼저 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">해당 글</a>을 읽어보시면서 AlexNet 구현체 및 개vs고양이 분류 데이터셋에 대한 준비를 미리 완료해 주시길 바랍니다.
  - 본 글에서 최적 hyperparameter 탐색을 수행하는 전체 구현체 코드는 *TODO: 링크 명시* 수아랩의 GitHub 저장소에서 자유롭게 확인하실 수 있습니다.
    - 전체 구현체 코드 원본에는 모든 주석이 (일반적인 관습에 맞춰) 영문으로 작성되어 있으나, 본 글에서는 원활한 설명을 위해 이들을 한국어로 번역하였습니다.

## 서론 

**Hyperparameter Optimization**이란, 학습을 수행하기 위해 사전에 설정해야 하는 값인 hyperparameter(하이퍼파라미터)의 최적값을 탐색하는 문제를 지칭합니다. 여기에서 hyperparameter의 최적값이란, 학습이 완료된 러닝 모델의 일반화 성능을 최고 수준으로 발휘하도록 하는 hyperparameter 값을 의미합니다.

딥러닝 모델을 학습하는 상황을 예로 들면, 학습률(learning rate), 미니배치 크기(minibatch size), L2 정규화 계수(L2 regularization coefficient) 등이 대표적인 hyperparameter라고 할 수 있습니다. 물론 앞서 소개한 것들은 엄밀히 말하면 학습 알고리즘 또는 정규화(regularization)와 관련된 hyperparameter들이며, 경우에 따라서는 딥러닝 모델의 구조를 결정하는 요소들(e.g. 층 수, 컨볼루션 필터 크기 등)도 hyperparameter로 간주하여 탐색의 대상으로 추가될 수 있습니다.

### Manual Search

딥러닝 모델을 한 번 이상 학습해 보신 분들이라면, 틀림없이 이러한 주요 hyperparameter들의 값을 결정하는 데 있어 많은 시행착오를 겪으셨던 경험이 있을 것입니다. 예를 들어 AlexNet 모델에 대한 구현을 완료했다고 하면, 보통은 원본 AlexNet 논문에서 소개된 hyperparameter를 그대로 가져와 학습에 적용하는 것으로 작업을 시작하는 경우가 대부분입니다. 하지만 대부분의 상황에서, 원본 AlexNet 논문에서 사용했던 데이터셋과 내가 사용하고자 하는 데이터셋이 서로 다르기 때문에, 원 논문에서 소개된 hyperparameter 값이 여러분들이 해결하고자 하는 문제에 한 방에 완벽하게 적용되는 경우는 잘 없습니다.

이러한 상황을 직면했을 시, 보통은 여러분들의 직관 또는 대중적으로 알려진 노하우 등에 의존하여, 다음으로 시도할 후보 hyperparameter 값을 선정하고, 이들을 사용하여 학습을 수행한 후 검증 데이터셋(validation set)에 대하여 측정한 성능 결과를 기록합니다. 이러한 과정을 몇 차례 거듭한 후, 맨 마지막 시점까지의 시도들 중 검증 데이터셋에 대하여 가장 높은 성능을 발휘했던 hyperparameter 값들을 선정, 최종 제출용 딥러닝 모델을 학습하는 방식을 채택해 왔을 것입니다. 이와 같이 최적 hyperparameter 값을 탐색하는 방법을 **Manual Search**라고도 합니다.

Manual Search는 Hyperparameter Optimization을 위한 가장 직관적인 방법이긴 하나, 몇 가지 문제가 있습니다. 첫째로, '최적의' hyperparameter를 찾을 때의 과정이, 다소 운(?)에 좌우된다는 점입니다. 그 예로, 여러분이 딥러닝 모델의 최적 학습률을 찾기 위해 Manual Search를 수행하는 과정을 묘사해 보도록 하겠습니다. 높은 확률로, 이 과정에는 시간 제한이 존재하며, 여러분들은 아래와 같이 생각하면서 대단히 조급해 하고 계실 가능성이 높습니다:

> 딥러닝 모델로 빨리 성능 뽑아야 하는데, 교수님 or 직장 상사는 계속 재촉하고, 시간은 없고.. 큰일났다 ㅠㅠ

{% include image.html name=page.name file="unluck-in-manual-search-process.gif" description="학습률에 대한 Manual Search 과정에서의 불운한 결과 예시" class="full-image" %}

여러분이 제한된 시간 안에 $$0.001$$, $$0.005$$, $$0.003$$, $$0.002$$, $$0.0035$$, $$0.0025$$까지 총 6가지 학습률 값을 순서대로 적용하여 딥러닝 모델을 학습하고 그 성능을 측정한 결과가 위 그림의 상단 결과와 같이 나왔고, 이에 따라 $$0.0025$$를 최종 학습률 값으로 채택하였다고 가정해 보겠습니다. 이 탐색 과정은 나름의 직관을 적용해서 매 학습 때마다 조심스럽게 진행한 것일 거고, 사람의 심리 상 이렇게 심혈을 기울인 과정을 통해 얻은 결과가 최선의 결과임을 부정하기는 대단히 힘들 것입니다.

그러나 실제로 '학습률에 따른 (미지의) 일반화 성능 함수'가 그 바로 하단과 같았다면 어떨까요? 실제로는 $$0.0025$$가 최적의 학습률 값이 아니었음에도 불구하고($$0.003$$과 $$0.0035$$ 사이의 값이 최적), 기존의 수동적인 탐색 과정에서 여러분의 조급함과 편견으로 인해 아쉬운 결과가 얻어진 것이라고 추측해볼 수 있습니다. 본의 아니게 여러분이 과거에 행하셨을 법한 과오를 지적한(?) 꼴이 됐지만, 이건 비단 여러분의 책임만은 아닙니다. 주관과 직관에 기반한 Manual Search를 진행할 경우, 위 사례에서 보여진 것처럼, 여러분들이 찾은 최적 hyperparameter 값이 '실제로도' 최적이라는 사실을 보장하기가 상대적으로 어렵다는 단점이 있습니다.

Manual Search의 두 번째 문제는, 한 번에 여러 종류의 hyperparameter들을 동시에 탐색하고자 할 시, 문제가 더욱 복잡해진다는 것입니다. 이를 가장 잘 보여줄 수 있는 예로 학습률-L2 정규화 계수 간의 관계가 있습니다.

$$
L(W) = \frac{1}{N} \sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda \cdot R(W)
$$

상기 손실 함수(loss function) 식에서 두 번째 항에 해당하는 것이 L2 정규화 항인데, 여기의 L2 정규화 계수인 $$\lambda$$의 값을 변화시키면 (딥러닝 모델의 전체 파라미터 $$W$$ 공간 상에서) 손실 함수 $$L(W)$$의 형태도 변화하게 됩니다. 이로 인해, 최적 성능을 발휘하도록 하는 최적 학습률의 값도 자연스럽게 변화할 것이라고 추측할 수 있습니다.

이와 같이 여러 종류의 hyperparameter들 중에는 서로 간의 상호 영향 관계를 나타내는 것들도 존재하기 때문에, 둘 이상의 hyperparameter들에 대한 탐색을 한 번에 진행할 시, 단일 hyperparameter 각각에 대하여 기존의 직관을 적용하기가 매우 어려워집니다.


### Grid Search vs. Random Search

Manual Search에 비해, Grid Search와 Random Search는 상대적으로 체계적인 방식으로 Hyperparameter Optimization을 수행하는 방법에 해당합니다. 

**Grid Search**는 탐색의 대상이 되는 특정 구간 내의 후보 hyperparameter 값들을 일정한 간격을 두고 선정하여, 이들 각각에 대하여 측정한 성능 결과를 기록한 뒤, 가장 높은 성능을 발휘했던 hyperparameter 값을 선정하는 방법입니다. 전체 탐색 대상 구간을 어떻게 설정할지, 간격의 길이는 어떻게 설정할지 등을 결정하는 데 있어 여전히 사람의 손이 필요하나, 앞선 Manual Search와 비교하면 좀 더 균등하고 전역적인 탐색이 가능하다는 장점이 있습니다. 반면 탐색 대상 hyperparameter의 개수를 한 번에 여러 종류로 가져갈수록, 전체 탐색 시간이 기하급수적으로 증가한다는 단점이 있습니다.

{% include image.html name=page.name file="grid-search-vs-random-search.png" description="Grid Search 결과와 Random Search 결과 비교 예시<br><small>\[Bergstra and Bengio(2012)\]</small>" class="full-image" %}

반면 **Random Search**는 Grid Search와 큰 맥락은 유사하나, 탐색 대상 구간 내의 후보 hyperparameter 값들을 랜덤 샘플링(sampling)을 통해 선정한다는 점이 다릅니다. Random Search는 Grid Search에 비해 불필요한 반복 수행 횟수를 대폭 줄이면서, 동시에 정해진 간격(grid) 사이에 위치한 값들에 대해서도 확률적으로 탐색이 가능하므로, 최적 hyperparameter 값을 더 빨리 찾을 수 있는 것으로 알려져 있습니다. 

그럼에도 불구하고, Random Search도 '여전히 약간의 불필요한 탐색을 반복하는 것 같다'는 느낌을 지우기 어려우실 것이라고 생각합니다. 왜냐하면 Grid Search와 Random Search 모두, 바로 다음 번 시도할 후보 hyperparameter 값을 선정하는 과정에서, 이전까지의 조사 과정에서 얻어진 hyperparameter 값들의 성능 결과에 대한 '사전 지식'이 전혀 반영되지 않았기 때문입니다(반면 Manual Search 과정에서는 사전 지식이 매 차례마다 여러분들의 은연 중에 잘 반영된 바 있습니다). 

매 회 새로운 hyperparameter 값에 대한 조사를 수행할 시 '사전 지식'을 충분히 반영하면서, 동시에 전체적인 탐색 과정을 체계적으로 수행할 수 있는 방법론으로, Bayesian Optimization을 들 수 있습니다.


## Bayesian Optimization

**Bayesian Optimization**은 본래, 어느 입력값 $$x$$를 받는 미지의 목적 함수(objective function) $$f$$를 상정하여, 그 함숫값 $$f(x)$$를 최대로 만드는 최적해 $$x^{*}$$를 찾는 것을 목적으로 합니다. 보통은 목적 함수의 표현식을 명시적으로 알지 못하면서(i.e. black-box function), 하나의 함숫값 $$f(x)$$를 계산하는 데 오랜 시간이 소요되는 경우를 가정합니다. 이러한 상황에서, 가능한 한 적은 수의 입력값 후보들에 대해서만 그 함숫값을 순차적으로 조사하여, $$f(x)$$를 최대로 만드는 최적해 $$x^{*}$$를 *빠르고 효과적으로* 찾는 것이 주요 목표라고 할 수 있습니다. 

이 때, 입력값 $$x$$를 '딥러닝 모델의 hyperparameter', 목적 함수의 함숫값 $$f(x)$$를 '해당 hyperparameter에 대한 딥러닝 모델의 성능 결과'로 간주하면, Bayesian Optimization 방법을 Hyperparameter Optimization에 그대로 적용할 수 있습니다. 이전까지의 조사 과정에서 얻어진 hyperparameter 값들의 성능 결과에 대한 '사전 지식'을 반영하여, 최종적으로 '최적 hyperparameter 값을 찾는 데 있어 가장 유용한 정보'를 가져다 줄 만한 hyperparameter 값 후보를 바로 다음 차례에 시도하는 것이, 전체 과정의 핵심이라고 할 수 있습니다. 이는 여러분이 은연 중에 수행했던 Manual Search 과정을 체계적으로 자동화한 것에 해당한다고 볼 수 있습니다.

Bayesian Optimization에는 두 가지 필수 요소가 존재합니다. 먼저 **Surrogate Model**은, 현재까지 조사된 입력값-함숫값 점들 $$(x_1, f(x_1)), ..., (x_t, f(x_t))$$를 바탕으로, 미지의 목적 함수의 형태에 대한 확률적인 추정을 수행하는 모델을 지칭합니다. 그리고 **Acquisition Function**은, 목적 함수에 대한 현재까지의 확률적 추정 결과를 바탕으로, '최적 입력값 $$x^{*}$$를 찾는 데 있어 가장 유용할 만한' 다음 입력값 후보 $$x_{t+1}$$을 추천해 주는 함수를 지칭합니다.

*TODO: Bayesian Optimization의 의사 코드(pseudo-code)*


### Surrogate Model

현재까지 조사된 입력값-함숫값 점들 $$(x_1, f(x_1)), ..., (x_t, f(x_t))$$를 바탕으로, 미지의 목적 함수의 대략적인 형태에 대한 확률적인 추정을 수행하는 모델을 Surrogate Model이라고 하였습니다. Surrogate Model로 가장 많이 사용되는 확률 모델(probabilistic model)이 **Gaussian Process(이하 GP)**입니다.

#### Gaussian Processes(GP)

GP는 (어느 특정 변수에 대한 확률 분포를 표현하는) 보통의 확률 모델과는 다르게, 모종의 *함수*들에 대한 확률 분포를 나타내기 위한 확률 모델이며, 그 구성 요소들 간의 결합 분포(joint distribution)가 *가우시안 분포(Gaussian distribution)*를 따른다는 특징이 있습니다. GP는 평균 함수 $$\mu$$와 공분산 함수 $$k$$를 사용하여 함수들에 대한 확률 분포를 표현합니다.

$$
f(x) \sim \mathcal{GP}(\mu(x), k(x, x')).
$$

GP를 제대로 이해하고 사용하려면 베이지안 확률론(Bayesian probability)에 대한 기본적인 이해와 더불어, 복잡해 보이는 확률적/선형대수적 수식을 이해할 수 있어야 합니다. 본 글에서는 이들에 대한 더 이상의 자세한 설명은 생략하며, GP의 동작 특징과 더불어 GP가 Hyperparameter Optimization을 위해 어떻게 사용될 수 있는지에 대해서만 초점을 맞추어 설명하고자 합니다.

현재까지 조사된 입력값-함숫값 점들 $$(x_1, f(x_1)), ..., (x_t, f(x_t))$$를 바탕으로, GP는 목적 함수에 대한 확률적 추정을 아래 그림과 같이 수행합니다.

{% include image.html name=page.name file="bayesian-optimization-procedure-example.png" description="GP를 사용한 Bayesian Optimization 진행 과정 예시 <br><small>(검은색 점선: 실제 목적 함수, 검은색 실선: 추정된 평균 함수, <br>파란색 음영: 추정된 표준편차, 검은색 점: 현재까지 조사된 입력값-함숫값 점, <br>하단의 녹색 실선: Acquisition Function)<br>\[Brochu et al.(2010)\]</small>" class="full-image" %}

위 그림에서 가로축을 입력값 $$x$$, 세로축을 함숫값 $$f(x)$$라고 보면, 검은색 실선으로 표시된 것이 (현재까지 조사된 점들 $$(x_1, f(x_1)), ..., (x_t, f(x_t))$$에 의거하여 추정된) 각 $$x$$ 위치 별 **'평균'** $$\mu(x)$$이며, 파란색 음영으로 표시된 것이 각 $$x$$ 위치 별 **'표준편차'** $$\sigma(x)$$에 해당합니다. $$\mu(x)$$의 경우 현재까지 조사된 점들 $$(x_1, f(x_1)), ..., (x_t, f(x_t))$$를 반드시 지나도록 그 형태가 결정되며, 조사된 점에서 가까운 위치에 해당할수록 $$\sigma(x)$$가 작고, 멀어질수록 $$\sigma(x)$$가 커지는 양상을 나타내고 있습니다. 이는 곧 **조사된 점으로부터 거리가 먼 $$x$$일수록, 이 지점에 대해 추정한 평균값의 '불확실성'이 크다**는 의미를 자연스럽게 내포하고 있습니다.

위 그림에서 $$t=2$$일 시에는, 조사된 입력값-함숫값 점이 2개에 불과하므로, 이 두 점에서 일정 수준 이상 떨어진 대부분의 영역에서 $$\sigma(x)$$가 큰 것을 관찰할 수 있습니다. 한편 $$t=3$$, $$t=4$$로 조사된 점의 개수가 점차적으로 늘어날수록, $$\sigma(x)$$가 큰 영역의 크기가 점차적으로 감소하며, 실제 목적 함수에 대한 추정 결과가 점차적으로 '압축'되는 양상을 보입니다. 즉, 조사된 점의 개수가 증가할수록 목적 함수의 추정 결과에 대한 '불확실성'이 감소되었다는 것을 보여주며, 그런 경향이 강해질수록 목적 함숫값을 최대로 만드는 입력값 $$x^{*}$$를 제대로 찾을 가능성이 계속해서 높아질 것이라고 짐작할 수 있습니다.

#### GP 외의 Surrogate Models

GP 외에도, 현재까지 조사된 입력값-함숫값 점들을 바탕으로 목적 함수 추정에 있어서의 '불확실성'을 커버할 수 있는 모델은 Surrogate Model로써 활용이 가능합니다. GP 외에 많이 사용되는 Surrogate model로는 Tree-structured Parzen Estimators(TPE), Random Forests, Deep Neural Networks 등이 있습니다. 

GP와 동일한 맥락에서, 이들 Surrogate Model들에 대한 깊은 이해가 없더라도, Bayesian Optimization의 큰 맥락을 이해하고 있다면 이와 관련된 라이브러리를 사용하여 Bayesian Optimization을 충분히 수행할 수 있습니다.


### Acquisition Function

Surrogate Model이 목적 함수에 대하여 확률적으로 추정한 현재까지의 결과를 바탕으로, 바로 다음 번에 함숫값을 조사할 입력값 후보 $$x_{t+1}$$을  추천해 주는 함수를 Acquisition Function이라고 하였습니다. 이 때 선정되는 $$x_{t+1}$$은, 결과적으로 목적 함수의 최적 입력값 $$x^{*}$$을 찾는 데 있어 '가장 유용할 만한' 것이라고 언급한 바 있습니다. 이 때의 '유용할 만하다'는 표현의 의미에 대하여 생각해 보도록 하겠습니다. 설명을 위해, GP를 사용한 목적 함수 추정 과정에서 $$t=2$$일 때의 상황을 나타내는 그림을 다시 가져왔습니다.

{% include image.html name=page.name file="bayesian-optimization-procedure-example-teq2.png" description="GP를 사용한 Bayesian Optimization 진행 과정 중 $$t=2$$일 때의 상황" class="full-image" %}

일단 현재까지 조사된 $$(x, f(x))$$ 점들이 두 개인데, 지금 상황에서 오로지 이들만을 놓고 생각해 보면, 두 점 중에서는 '함숫값이 더 큰 점(그림 상에서 오른쪽에 위치한 점) 근방에서 실제 최적 입력값 $$x^{*}$$를 찾을 가능성이 높을 것이다'라고 예측하는 것은 어느 정도 그럴싸한 예측이라고 할 수 있습니다. 자연히, 현재까지 조사된 점들 중 함숫값이 최대인 점 근방을 그 다음 차례에 시도하는 것이 나름대로 합리적인 전략이라고 할 수 있습니다. 이러한 행위를 정식 용어로는 '**exploitation(착취, 수탈)**'이라고 합니다.

이번에는 관점을 달리하여 생각해 보겠습니다. 현재까지 조사된 두 점 사이에 위치하였으면서 표준편차(=불확실성) $$\sigma(x)$$가 큰 영역의 경우, 이 부분의 추정된 평균 함숫값이 실제 목적 함숫값과 유사할 것이라고 장담하기 매우 어려울 것임을 직관적으로 느낄 수 있습니다. 그렇게 보면, '불확실한 영역에 최적 입력값 $$x^{*}$$이 존재할 가능성이 있으므로, 이 부분을 추가로 탐색해야 한다'고 생각하는 것이 어느 정도 그럴싸한 판단이라고 할 수 있으며, 이에 따라 현재까지 추정된 목적 함수 상에서 표준편차가 최대인 점 근방을 그 다음 차례에 시도하는 것 또한 나름대로 합리적인 전략입니다. 이러한 행위를 정식 용어로는 '**exploration(탐색)**'이라고 합니다.

Exploration 전략과 exploitation 전략 모두, 최적 입력값 $$x^{*}$$를 효과적으로 찾는 데 있어 균등하게 중요한 접근 전략이라고 할 수 있으나, 문제는 두 전략의 성격이 서로 trade-off 관계에 있다는 데에 있습니다. 따라서 이러한 exploration-exploitation 간의 상대적 강도를 적절하게 조절하는 것이, 실제 목적 함수에 대한 성공적인 최적 입력값 탐색에 매우 중요하다고 할 수 있습니다.

#### Expected Improvement(EI)

**Expected Improvement(이하 EI)** 함수는 이러한 exploration 전략 및 exploitation 전략 모두를 내재적으로 일정 수준 포함하도록 설계된 것으로, Acquisition Function으로 가장 많이 사용됩니다. EI는 현재까지 추정된 목적 함수를 바탕으로, 어느 후보 입력값 $$x$$에 대하여 '현재까지 조사된 점들의 함숫값 $$f(x_1), ..., f(x_t)$$ 중 최대 함숫값 $$f(x^{+}) = \max_{i} f(x_i)$$보다 더 큰 함숫값을 도출할 확률(Probability of Improvement; 이하 *PI*)' 및 '그 함숫값과 $$f(x^{+})$$ 간의 *차이값*(magnitude)'을 종합적으로 고려하여, 해당 입력값 $$x$$의 '유용성'을 나타내는 숫자를 출력합니다. 이 때, PI의 개념을 이해해 보기 위해 아래 그림을 살펴보도록 하겠습니다.

{% include image.html name=page.name file="probability-of-improvement-in-gaussian-process-example.png" description="'최대 함숫값 $$f(x^{+})$$보다 더 큰 함숫값을 도출할 확률(PI)'에 대한 시각화 예시<br><small>(세로 방향 점선: 입력값 $$x_1$$, $$x_2$$, $$x_3$$에서의 함숫값 $$f(x_1)$$, $$f(x_2)$$, $$f(x_3)$$ 각각에 대한 확률 분포, <br>초록색 음영: $$f(x_3)$$의 확률 분포 상에서, 그 값이 $$f(x^{+})$$보다 큰 영역)<br>[Brochu et al.(2010)]</small>" class="large-image" %}

위 그림에서 현재까지 조사된 점들의 함숫값 중 최대 함숫값 $$f(x^{+})$$는 맨 오른쪽에 위치한 점에서 발생하였습니다. 이 때, 그보다 더 오른쪽에 위치한 후보 입력값 $$x_3$$에 대하여, 확률적 추정 결과에 의거하여 $$f(x_3)$$의 (세로축 값에 따른) 확률 분포를 그림과 같이 기울어진 가우시안 분포 형태로 나타내 볼 수 있습니다. 

한편, $$f(x_3)$$의 확률 분포 상에서 $$f(x^{+})$$보다 큰 값에 해당하는 영역을 그림 상에서 초록색 음영으로 표시하였습니다. 이 영역의 크기가 클 수록, '$$f(x_3)$$이 $$f(x^{+})$$보다 클 확률이 높다'는 것을 나타내며, 이는 곧 **다음 입력값으로 $$x_3$$을 채택했을 시 기존 점들보다 더 큰 함숫값을 얻을 가능성이 높을 것**이라는 결론으로 연결되며, 목적 함수의 최적 입력값 $$x^{*}$$을 찾는 데 있어 $$x_3$$이 '가장 유용할 만한' 후보라고 판단할 수 있습니다.

이렇게 입력값 $$x_3$$에 대하여 계산한 PI 값에, 함숫값 $$f(x_3)$$에 대한 평균 $$\mu(x_3)$$과 $$f(x^{+})$$ 간의 차이값 $$f(x_3)-f(x^{+})$$만큼을 가중하여, $$x_3$$에 대한 EI 값을 최종적으로 계산합니다. '기존 점들보다 더 큰 함숫값을 얻을 가능성이 높은 점을 찾는 것'도 중요하지만, '**그러한 가능성이 존재한다면, 실제로 그 값이 얼마나 더 큰가**'도 중요한 고려 대상이기 때문에, 이를 반영하기 위한 계산 방식이라고 이해하면 되겠습니다.

참고용으로, EI의 계산식을 구체적으로 정리하면 아래와 같습니다. 아래 식에서 $$\Phi$$와 $$\phi$$는 각각 표준정규분포의 누적분포함수(CDF)와 확률분포함수(PDF)를 나타내며, $$\xi$$는 exploration과 exploitation 간의 상대적 강도를 조절해 주는 파라미터입니다.

$$
\begin{align}
EI(x) & = \mathbb{E} [\max (f(x) - f(x^{+}), 0)] \\
      & = 
\begin{cases}
		(\mu(\boldsymbol{x}) - f(\boldsymbol{x}^{+})-\xi)\Phi(Z) + \sigma(\boldsymbol{x})\phi(Z) & \text{if}\ \sigma(\boldsymbol{x}) > 0 \\
    0 & \text{if}\ \sigma(\boldsymbol{x}) = 0 
\end{cases}
\end{align}
$$

$$
Z = 
\begin{cases}
    \frac{\mu(\boldsymbol{x})-f(\boldsymbol{x}^{+})-\xi}{\sigma(\boldsymbol{x})} & \text{if}\ \sigma(\boldsymbol{x}) > 0 \\ 
    0 & \text{if}\ \sigma(\boldsymbol{x}) = 0
\end{cases}
$$

GP를 사용한 목적 함수 추정 과정의 $$t=4$$일 때의 상황에서, 각 입력값 $$x$$ 별 EI의 값 $$EI(x)$$를 계산한 결과를 도시하면, 아래 그림의 하단에 있는 초록색 실선과 같이 나타납니다. 

{% include image.html name=page.name file="bayesian-optimization-procedure-example-teq4.png" description="GP를 사용한 Bayesian Optimization 진행 과정 중 $$t=4$$일 때의 상황" class="full-image" %}

실제로 현재까지 조사된 점들 중 최대 함숫값을 가지는 점 $$x^{+}$$ 주변에서 EI 값이 크고(exploitation 전략), 그와 동시에 현재까지 추정된 목적 함수 상에서 표준편차 $$\sigma(x)$$가 최대인 점 주변에서도 EI 값이 큰(exploration 전략) 것을 그림 상에서 동시에 관찰할 수 있습니다.

#### EI 외의 Acquisition Functions

EI보다 이른 시기에 먼저 제안되었으면서, EI의 고려 대상들 중 '현재까지 조사된 점들의 함숫값 중 최대 함숫값보다 더 큰 함숫값을 도출할 확률'만을 반영한 Acquisition Function을 Probability of Improvement(PI)라고 합니다. 그 외에도 많이 사용되는 Acquisition Function으로는 Upper Confidence Bound(UCB), Entropy Search(ES) 등이 있습니다. 


## 딥러닝 모델의 hyperparameter 탐색을 위한 Bayesian Optimization 수행 과정

지금까지 Bayesian Optimization의 필수 요소 및 이들의 기본적인 작동 방식을 파악해 보았습니다. 이제 실제로 딥러닝 모델의 hyperparameter를 탐색할 시 Bayesian Optimization이 적용되는 시나리오에 대하여 좀 더 구체적으로 가시화하여 보여 드리도록 하겠습니다. 설명의 편의를 위해 학습률(learning rate) 하나만을 탐색 대상 hyperparameter로 놓고 서술하였습니다.

*TODO: 아래 과정을 나타내는 gif 그림 추가*

1. 입력값, 목적 함수 및 그 외 설정값들을 정의함.
  - 입력값 $$x$$: 학습률
  - 목적 함수 $$f(x)$$: 설정한 학습률을 적용하여 학습한 딥러닝 모델의 검증 데이터셋에 대한 성능 결과 수치(e.g. 정확도)
  - 입력값 $$x$$의 탐색 대상 구간을 설정함: $$(a, b)$$.
  - 맨 처음에 조사할 입력값-함숫값 점들의 갯수: $$n$$
  - 조사할 입력값-함숫값 점들의 최대 갯수: $$N$$
2. 구간 $$(a, b)$$ 내에서 처음 $$n$$개의 입력값을 랜덤하게 샘플링함.
3. 선택한 $$n$$개의 입력값 $$x_1, x_2, ..., x_n$$을 각각 학습률 값으로 설정하여 딥러닝 모델을 학습한 뒤, 검증 데이터셋을 사용하여 학습이 완료된 모델의 성능 결과 수치를 계산하고 이들을 각각 $$f(x_1), f(x_2), ..., f(x_n)$$ 값으로 간주함.
4. 조사된 입력값-함숫값 점들이 총 $$N$$개에 도달할 때까지, 아래의 과정을 반복적으로 수행함($$t=n, n+1, ..., N$$). 조사된 점들의 갯수가 $$N$$개에 도달한 순간, GP를 사용한 확률적 추정까지만 수행하고, 나머지 과정을 중단함.
    - 현재까지 조사된 입력값-함숫값 점들 $$(x_1, f(x_1)), (x_2, f(x_2)), ..., (x_t, f(x_t))$$을 바탕으로, GP를 사용하여 미지의 목적 함수에 대한 확률적인 추정을 수행함.
    - 확률적 추정 결과에 대하여, 입력값 구간 $$(a, b)$$ 내에서 EI의 값을 계산하고, 그 값이 가장 큰 점을 다음 입력값 후보 $$x_{t+1}$$로 선정함.
    - 다음 입력값 후보 $$x_{t+1}$$을 학습률 값으로 설정하여 딥러닝 모델을 학습한 뒤, 검증 데이터셋을 사용하여 학습이 완료된 모델의 성능 결과 수치를 계산하고 이를 $$f(x_{t+1})$$ 값으로 간주하고, 새로운 점 $$(x_{t+1}, f(x_{t+1}))$$을 기존 입력값-함숫값 점들의 모음에 추가함.
5. 총 $$N$$개의 입력값-함숫값 점들로 추정된 목적 함수 결과물을 바탕으로, 평균 $$\mu(x)$$을 최대로 만드는 최적해 $$x^{*}$$를 최종 선택함.


## 결론

\*다음 회에서는, 지금까지의 이해를 바탕으로 하여 실제 Bayesian Optimization을 위한 Python 라이브러리인 bayesian-optimization을 사용하여 간단한 예시 함수의 최적해를 탐색하는 과정을 먼저 소개하고, 실제 딥러닝 모델의 hyperparameter 탐색 과정을 여러분들께 안내해 드리고자 합니다.

(다음 포스팅 보기: COMING SOON)


## References

- Shahriari et al., Taking the human out of the loop: A review of bayesian optimization.
    - <a href="https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf" target="_blank">Shahriari, Bobak, et al. "Taking the human out of the loop: A review of bayesian optimization." Proceedings of the IEEE 104.1 (2016): 148-175.</a>
- Brochu et al., A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning.
    - <a href="https://arxiv.org/pdf/1012.2599.pdf?bcsi_scan_dd0fad490e5fad80=fwQqmV5CfHDAMm8dFLewPK+h1WGiAAAAkj1aUQ%3D%3D&bcsi_scan_filename=1012.2599.pdf&utm_content=buffered388&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer" target="_blank">Brochu, Eric, Vlad M. Cora, and Nando De Freitas. "A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning." arXiv preprint arXiv:1012.2599 (2010).</a>
- Bengio et al., Practical recommendations for gradient-based training of deep architectures.
    - <a href="https://arxiv.org/pdf/1206.5533.pdf" target="_blank">Bengio, Yoshua. "Practical recommendations for gradient-based training of deep architectures." Neural networks: Tricks of the trade. Springer, Berlin, Heidelberg, 2012. 437-478.</a>
- Goodfellow et al., Deep learning.
    - <a href="https://www.deeplearningbook.org/" target="_blank">Goodfellow, Ian, et al. Deep learning. Vol. 1. Cambridge: MIT press, 2016.</a>
- Bergstra and Bengio, Random search for hyper-parameter optimization.
    - <a href="http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf" target="_blank">Bergstra, James, and Yoshua Bengio. "Random search for hyper-parameter optimization." Journal of Machine Learning Research 13.Feb (2012): 281-305.</a>
- Fernando Nogueira, bayesian-optimization: A Python implementation of global optimization with gaussian processes.
    - <a href="https://github.com/fmfn/BayesianOptimization" target="_blank">https://github.com/fmfn/BayesianOptimization</a>
