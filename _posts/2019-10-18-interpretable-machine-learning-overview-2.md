---
layout: post
title: "Interpretable Machine Learning 개요: (2) 이미지 인식 문제에서의 딥러닝 모델의 주요 해석 방법"
date: 2019-10-18 09:00:00 +0900
author: kilho_kim
categories: [Introduction]
tags: [interpretable machine learning, interpretability, explainable artificial intelligence]
comments: true
name: interpretable-machine-learning-overview-2
---

앞선 글에서 머신러닝 모델에 대한 해석력 확보를 위한 Interpretable Machine Learning(이하 IML)의 개요를 다뤘습니다. 이번 글에서는 IML에 대한 지금까지의 이해를 바탕으로, 많은 분들이 관심을 가지고 계실 딥러닝 모델에 대한 주요 IML 방법론들에 대해 알아보고자 합니다.

본래 앞서 소개했던 Post-hoc(Model-agnostic) 계열의 IML 방법론들은 전통적인 머신러닝 모델뿐만 아니라 신경망(neural network)에도 적용될 수 있도록 범용적으로 디자인되어 있었습니다. 그런데 오늘날 딥러닝 모델이 다양한 문제들에 대하여 특출나게 높은 성능을 발휘하게 되면서 다른 종류의 모델보다 집중적인 조명을 받게 되었고, 자연스럽게 깊은 신경망 자체에 대한 해석에 초점을 맞추는 연구들이 하나둘씩 늘어나기 시작했습니다. 이에 따라 2010년대 중반부터는 깊은 신경망의 구조적 특징을 고려한 IML 방법론들이, 전통적인 IML 방법론 계열들과는 별도의 독자적인 흐름을 형성하는 경향을 보여 왔습니다. 특히, 결과 해석에 있어서의 상대적인 용이성 때문에, 이미지 인식(image recognition) 문제에 적용된 딥러닝 모델에 대한 연구가 활발하게 진행되어 왔습니다.

앞선 글에서 머신러닝 전반에 대한 IML 방법론들의 거시적인 개요에 대해 소개해 드렸다면, 본 글에서는 관심의 대상을 딥러닝으로 좁혀 이미지 인식 문제에서의 딥러닝 모델을 위해 특화된 IML 방법론에 대하여 집중적으로 소개해 드리고자 합니다. 

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - 딥러닝 알고리즘의 기본 구동 원리 및 딥러닝 모델(특히 컨볼루션 신경망)에 대한 기초적인 내용들을 이해하고 계신 분들
- 현재도 딥러닝 모델에 대한 IML 방법론들의 분류 방식이 학계에서 다양하게 존재하며, 아직까지 하나로 완벽하게 정립되지는 않은 상태입니다. 본 글에서는 Google Brain 팀에서 포스팅한 블로그 글 <a href="https://distill.pub/2017/feature-visualization" target="_blank">(1)</a>, <a href="https://distill.pub/2018/building-blocks" target="_blank">(2)</a> <small>(Christopher Olah et al.)</small>에서 소개한 분류 방식을 주로 참조하되, 원활한 내용 전개를 위해 약간의 변형을 가하였음을 알려드립니다.


## Activation/Weight Visualization

오늘날 이미지 인식 문제에 사용되는 전형적인 딥러닝 모델인 <a href="{{ site.url }}{% post_url 2017-10-10-what-is-deep-learning-1 %}#컨볼루션-신경망" target="_blank">컨볼루션 신경망(CNN: convolutional neural network)</a>을 단순화시키면 아래 그림과 같이 나타낼 수 있습니다. 어느 이미지가 입력되면, 사전에 학습된 복수 개의 layer(층)들을 거치면서 feature extraction(요인 추출)을 수행하여 feature map(요인 맵)들이 생성되고, 이것이 뒤따르는 복수 개의 layer들을 거쳐 확률 형태의 prediction(예측) 결과를 산출하는 구조입니다.

{% include image.html name=page.name file="image-features-prediction-diagram.png" description="컨볼루션 신경망의 기본 구성" class="large-image" %}

컨볼루션 신경망이 특정 예측 결과를 도출하게 된 근거를 해석하기 위해 가장 직관적으로 생각할 수 있는 방법으로, 중간 계산 과정에서 생성된 feature map 각각을 들여다 보는 방법이 있습니다. 보통 일반적인 머신러닝 모델에서 feature를 '들여다 본다' 함은, feature로 계산된 값의 크기를 비교하고 통계적 분포 등을 조사하는 것이 될 것입니다. 그러나 컨볼루션 신경망의 feature map은 그 특유의 2차원적 구조 때문에 이를 마치 이미지처럼 그려볼 수 있다는 장점이 있습니다. 
 
{% include image.html name=page.name file="activation-visualization-example.jpeg" description="컨볼루션 신경망의 각 layer에서의 activation visualization 결과 예시 <small>(Andrej Karpathy)</small>" class="full-image" %}

위 그림은 3개 layer들로 구성된 아주 간단한 컨볼루션 신경망을 학습한 뒤, 여기에 자동차가 포함된 이미지를 입력했을 때의 각 layer에서의 feature map들을 이미지 형태로 표시한 결과를 보여주고 있습니다. 특정 이미지를 입력했을 때의 각 feature map이 '활성화된(activation)' 정도를 보여준다는 맥락에서, 이러한 해석 방법을 보통 '**Activation Visualization**(활성값 시각화)'이라고 부릅니다. Activation Visualization을 통해, 여러분들은 학습이 완료된 컨볼루션 신경망의 각 layer에서 어떤 feature map이 이미지 상의 어떠한 특징들을 커버하고 있는지 추측할 수 있습니다. 이를테면 아래 그림과 같이, 사람 또는 동물의 얼굴이 포함되어 있는 이미지들을 입력할 때, 공통적으로 최대로 활성화되는 feature map이 무엇인지 조사하고, 이들을 시각화함으로써 해당 feature map이 실제로 '얼굴'을 커버하는 경향을 보이는지 확인할 수 있습니다.

{% include image.html name=page.name file="face-activation-example.png" description="사람 또는 동물의 '얼굴'을 커버하는 feature map의 Activation Visualization 결과 예시 <small>(Jason Yosinski et al.)</small>" class="large-image" %}

Activation Visualization의 필수 조건을 꼽는다면, 해석 대상이 되는 컨볼루션 신경망에 반드시 예시 이미지들을 입력해 준 뒤 그 결과의 경향성을 관찰해야 한다는 점이 있습니다. 이러한 조건을 회피하기 위한 방법으로, 학습된 컨볼루션 신경망이 보유한 weights(가중치, 필터) 자체를 그대로 시각화해볼 수 있습니다. 컨볼루션 신경망의 weights 또한 2차원적 구조를 지니기 때문에, feature map처럼 이미지 형태로 그려볼 수 있습니다. 이를 보통 '**Weight Visualization**(가중치 시각화)'이라고 부릅니다.

{% include image.html name=page.name file="weight-visualization-example.png" description="컨볼루션 신경망의 1, 2, 3번째 컨볼루션 layer에서의 weight visualization 결과 예시<br><small>(Jost Tobias Springenberg et al.)</small>" class="full-image" %}

다만, 여러분이 위 예시 그림을 보면서도 느끼시겠지만, Weight Visualization 결과만을 관찰하는 것만으로는 신경망의 각 weights가 이미지 상의 어떤 시각적 특징을 커버하는지 직관적으로 이해하기가 어렵습니다. 관찰 대상이 되는 layer을 상위 레벨로 이동할 수록 더 많은 필터들이 등장하기 때문에, weights에 대한 해석의 난해함은 점점 심해지는 경향을 보입니다. 그렇기 때문에 보통 Weight Visualization의 경우 현재 학습 중인 컨볼루션 신경망의 상태를 점검하기 위한 목적 정도로만 쓰이는 경우가 많으며, 그로부터 그 이상의 시사점을 확인하기는 현실적으로 어렵다고 볼 수 있습니다.


## Activation Maximization

Activation/Weight Visualization 이후로, 각 feature map이 커버하는 시각적 특징이 정확히 무엇인지 더욱 효과적으로 찾고 가시화하고자 하는 시도가 자연스럽게 늘어나기 시작하였습니다. 그 중 하나로, 컨볼루션 신경망 상의 어느 타겟 출력값을 하나 고정해 놓고, 이를 최대로 활성화하는 입력 이미지를 찾거나 생성하는 방법을 생각해볼 수 있는데, 이를 '**Activation Maximization**(활성값 최대화)'이라고 부릅니다.

{% include image.html name=page.name file="activation-maximization-concept.png" description="Activation Maximization의 기본 컨셉<br><small>('argmax': 'TARGET'을 최대로 활성화하는 'INPUT' 탐색 또는 생성)</small>" class="large-image" %}

예를 들어, 위 그림과 같이 어느 feature map 상의 특정 neuron(뉴런)을 조사 타겟으로 설정하고, 이를 *최대 수준으로 활성화시키는* 입력 이미지의 형태를 조사하는 것이 가장 대표적인 Activation Maximization이라고 할 수 있습니다. 이 때, feature map 상의 특정 neuron 외에도 feature map(=channel) 또는 복수 개의 feature map들을 포괄하는 하나의 layer 등도 타겟으로 설정할 수 있으며, 최종 prediction layer의 특정 logit(로짓) 또한 타겟으로 설정할 수 있습니다. 

{% include image.html name=page.name file="types-of-target-activation.png" description="컨볼루션 신경망 상의 타겟 출력값 후보 <small>(Christopher Olah et al.)</small>" class="large-image" %}

### Maximally Activating Images

Activation Maximization을 위한 나이브한 방법으로, 컨볼루션 신경망 상의 타겟 출력값을 최대로 활성화하는 입력 이미지들을 현재 가지고 있는 데이터셋 상에서 탐색할 수 있는데, 해당 이미지들을 여기에서는 '**Maximally Activating Images**(활성값 최대화 이미지)'라고 부르겠습니다. 예를 들어, 아래 그림과 같이 타겟 출력값을 어느 하나의 feature map으로 설정할 경우, 데이터셋 상의 이미지들을 하나씩 컨볼루션 신경망에 입력하면서 그 과정에서 해당 feature map을 최대로 활성화시켰던 이미지가 무엇인지를 탐색합니다.

{% include image.html name=page.name file="maximally-activating-images-concept.png" description="Maximally Activating Images 탐색 예시<br><small>('argmax': 'TARGET'을 최대로 활성화하는 'INPUT'을 데이터셋 상에서 탐색)</small>" class="full-image" %}

Maximally Activating Images는 각 feature map을 최대로 활성화시키는 입력 이미지를 실제 데이터셋으로부터 탐색한다는 점에서, feature map이 커버하는 현실 이미지 상의 시각적 특징을 좀 더 직접적으로 파악하는 데 도움을 줍니다. 보통은 상위 몇 개의 Maximally Activating Images를 일단 뽑아낸 뒤 이들 각각에 대하여 Activation Visualization을 추가로 수행함으로써, 해당 feature map이 Maximally Activating Images 상의 어느 부분에 초점을 맞추는지 보다 면밀하게 확인할 수 있습니다. 그 결과, 특정 layer 내 각 feature map이 커버하는 현실 이미지 상의 시각적 특징들(e.g. 검정색 원형 부분, 물체의 곡선 경계, 알파벳 글자 등)을 아래 그림과 같이 patch(패치) 형태로 자세히 확인해볼 수 있습니다.

{% include image.html name=page.name file="maximally-activating-patches-example.png" description="컨볼루션 신경망의 6, 9번째 컨볼루션 layer 내 각 feature map의 상위 10개 Maximally Activating *Patches* 예시 <small>(Jost Tobias Springenberg et al.)</small><br><small>(각 행은 하나의 feature map에 대응되며, 좌측부터 활성값의 내림차순으로 상위 10개 이미지를 나타냄)</small>" class="full-image" %}

### Maximization by Optimization

Maximally Activating Images는 feature map 등이 커버하는 현실적 특징을 파악하기에는 용이하나, 그 탐색 범위가 현재 보유한 데이터셋 상의 이미지들로 한정된다는 약점이 있습니다. 이에 따라 만약 현재 보유하고 있는 이미지 수가 매우 적다면, Maximally Activating Images를 탐색하였을 시의 효용이 그다지 크지 않을 것이라고 예상할 수 있습니다.

보유한 데이터셋에 의존하지 않고 feature map 등이 커버하는 현실적 특징을 좀 더 직접적으로 조사하고자, gradient ascent(경사 상승법)에 기반한 optimization(최적화)을 통해 타겟 출력값을 최대로 활성화하는 입력 이미지를 직접 *생성*하는 접근이 시도되었습니다. 이러한 방법을 여기에서는 '**Maximization by Optimization**(최적화 기반 최대화)'이라고 부르겠습니다.  <small>(엄밀하게는, 보통 Activation Maximization이라고 하면, 이 Maximization by Optimization 방법을 의미합니다. Christopher Olah et al. 에서는 이를 '<a href="https://distill.pub/2017/feature-visualization" target="_blank">Feature Visualization</a>'이라고 표현하기도 합니다.)</small>

{% include image.html name=page.name file="maximization-by-optimization-concept.png" description="Maximization by Optimization 예시<br><small>('argmax': 'TARGET'을 최대로 활성화하는 'INPUT'을 생성)</small>" class="full-image" %}

예를 들어, 랜덤 노이즈(random noise) 형태의 이미지 $$X_0$$에서 출발하여, 위 그림과 같이 특정한 하나의 feature map $$f_k$$를 타겟 출력값으로 지정했다고 가정하겠습니다. 현재 이미지 $$X_0$$를 기준으로 feature map $$f_k$$의 이미지 $$X$$에 대한 gradient $$\partial f_{\cdot,\cdot,k} / \partial X$$를 계산하여 이를 현재 이미지에 더해주면, 기존보다 해당 feature map을 더 강하게 활성화시키는 새로운 이미지 $$X_1$$을 얻을 수 있습니다. 매 반복 회차 $$t$$마다 아래의 수식에 따라 이러한 과정을 반복하면서, 전체 반복 횟수 $$T$$를 충분히 많이 가져가면, 해당 feature map을 '최대로 활성화시키는 이미지'를 얻을 수 있습니다.

$$
X_{t+1} = X_t + \alpha \cdot \frac{\partial f_{\cdot,\cdot,k}}{\partial X}
$$

아래의 예시 그림들은 타겟 출력값을 서로 다른 feature map으로 설정했을 시 Maximization by Optimization 결과가 달라지는 것을 잘 보여주고 있습니다. 대체로 앞쪽 layer에 위치한 feature map들의 경우 단순하고 반복적인 패턴(edges, textures)을 커버하는 경향을 보이며, 뒷쪽 layer에 위치한 feature map들의 경우 그보다는 좀 더 복잡한 무늬, 사물의 일부분 또는 전체를 커버하는 경향을 보임을 확인할 수 있습니다.

{% include image.html name=page.name file="maximization-by-optimization-on-feature-map-examples.png" description="타겟 출력값을 서로 다른 feature map으로 설정했을 시의 서로 다른 Maximization by Optimization 수행 결과 예시 <small>(Christopher Olah et al.)</small>" class="full-image" %}

만일 Maximization by Optimization의 타겟 출력값을 feature map 대신 layer로 설정할 경우, 아래의 예시 그림들과 같이 상당히 드라마틱한 결과를 얻을 수 있습니다. 이들이 마치 꿈 속에서만 등장할 것 같은 생소한 인상을 주었기 때문에, 연구자들은 여기에 '**DeepDream**'이라는 이름을 붙였습니다. 그림에서도 확인하실 수 있듯이 layer 상의 feature map들에서 커버하는 패턴 또는 모양이 하나로 융합된 듯한 결과물을 생산해 내는데, 해석 가능성의 측면에서 봤을 땐 feature map 각각에 대한 관찰 결과에 비해 다소 난해한 듯한 특징을 보여주고 있습니다.

{% include image.html name=page.name file="deepdream-result-examples.png" description="타겟 출력값을 서로 다른 layer로 설정했을 시의 서로 다른 Maximization by Optimization(DeepDream) 수행 결과 예시 <small>(DeepDreaming with TensorFlow로 저자가 직접 생성한 결과물)</small>" class="full-image" %}

한편, Maximization by Optimization의 타겟 출력값을 최종 prediction layer의 logit으로 설정할 경우, 아래의 예시 그림들과 같이 좀 더 온전한 사물에 가까운 결과를 확인할 수 있습니다. 해당 logit에 대응되는 클래스(class)를 대표하는 '전형적인' 사물을 만들어 냈다고 할 수 있습니다. 이에 대한 관찰을 통해, 각 클래스에 대하여 주어진 학습 데이터셋으로부터 컨볼루션 신경망이 학습을 수행한 결과가 대략 어떤 형태인지 집약적으로 확인해 볼 수 있습니다.

{% include image.html name=page.name file="maximization-by-optimization-on-logit-examples.png" description="타겟 출력값을 logit으로 설정했을 시의 서로 다른 Maximization by Optimization 수행 결과 예시 <small>(Anh Nguyen et al.)</small>" class="full-image" %}


## Attribution

지금까지 살펴본 Activation Maximization 방법의 경우, 대부분 컨볼루션 신경망 중간에서 얻어지는 feature 또는 logit 등에 초점을 맞추어 그것들 각각이 이미지 상의 어떤 시각적 특징을 커버하는지 분석하는 것을 목표로 하였습니다. 즉 이는 학습이 완료된 컨볼루션 신경망을 구성하는 요소들 자체의 '일반적 행동 방식에 대한 이해'의 목적이 강하다고 볼 수 있습니다.

반면 지금부터 소개할 '**Attribution**(귀착, 귀속)'의 경우, 어떤 입력 이미지에 대한 컨볼루션 신경망의 예측 결과가 이미지 상의 어느 부분에 기인하였는지 찾기 위한 방법에 해당하며, 이미지 인식 문제에서의 딥러닝 모델의 예측 결과를 '설명(explanation)'하는 것을 목적으로 합니다.

사실 앞서 살펴보았던 Activation Visualization도 컨볼루션 신경망의 예측 결과에 대한 Attribution의 수단으로 활용될 수는 있으나, 아래 그림과 같이 많은 수의 feature map들을 매번 동시적으로 확인해야 한다는 점에서 그다지 매력적이지는 못합니다. 

{% include image.html name=page.name file="deep-visualization-example.jpg" description="Activation Visualization을 통한 예측 결과 설명 예시 <small>(Jason Yosinski et al.)</small><br><small>(좌측 상단: 입력 이미지, 중앙: 5번째 컨볼루션 layer의 모든 feature map들에 대한 시각화 결과)</small>" class="full-image" %}

### Saliency Map

컨볼루션 신경망의 Attribution을 보여주기 위한 대표적인 수단이 '**Saliency Map**(현저성 맵)'입니다. 보통 Saliency Map은 이미지 상의 두드러진 부분을 지칭하나, 컨볼루션 신경망의 예측 결과에 대한 설명의 맥락에서는, 예측 결과를 이끌어낸 이미지 상의 주요한 부분을 표현하기 위한 목적으로 생성됩니다.

컨볼루션 신경망의 예측 결과로부터 Saliency Map을 도출하기 위한 가장 간단한 방법은, 예측 클래스 logit $$y_c$$의 입력 이미지 $$X$$에 대한 gradient $$\partial y_c / \partial X$$를 계산하는 것입니다. 마치 앞서 소개했던 Maximization by Optimization과 유사해 보일 것인데, Maximization by Optimization이 랜덤한 이미지에서 출발하여 feature map의 gradient를 반복적으로 더해주는 gradient ascent를 통해 가상의 이미지를 생성하였다면, Saliency Map의 경우 실제 입력 이미지에 대한 예측 클래스 logit의 gradient를 한 번만 계산하여 이를 그대로 활용한다는 점이 차이라고 할 수 있겠습니다.

{% include image.html name=page.name file="saliency-map-with-gradient-concept.png" description="예측 클래스에 대한 gradient 계산을 통해 얻어진 Saliency Map 예시" class="full-image" %}

Saliency Map을 관찰함으로써, 컨볼루션 신경망의 특정 예측 결과가 이미지 상의 어느 부분에 기인하였는지 아래 그림과 같이 확인할 수 있습니다. 뿐만 아니라, 이렇게 얻어진 Saliency Map을 적절하게 가공하여, 이를 <a href="{{ site.url }}{% post_url 2017-11-29-image-recognition-overview-2 %}#segmentation" target="_blank">Segmentation 문제</a>에 적용하는 시도도 이루어지고 있습니다.

{% include image.html name=page.name file="saliency-map-examples.png" description="컨볼루션 신경망의 예측 결과에 대한 Saliency Map 도출 결과 예시 <small>(Karen Simonyan et al.)</small>" class="large-image" %}

### Class Activation Map

Attribution을 위해 Saliency Map 외에 많이 사용되는 또 다른 수단으로 '**Class Activation Map**(클래스 활성화 맵)'이 있습니다. 이는 최종 prediction layer 직전에 위치한 layer의 각 feature map에 대하여 *global average pooling(GAP)*을 수행하도록 설계된 컨볼루션 신경망에 대하여 범용적으로 적용할 수 있는 Attribution 방법에 해당합니다.

앞서 보았던 Activation Visualization이 feature map의 시각화 결과를 각각 시각화하는 방법이었다면, Class Activation Map은 prediction layer 직전의 weights를 사용하여 해당 feature map들의 가중합을 계산한 결과물만을 시각화함으로써, 특정 예측 클래스에 대한 전체 feature map들의 '평균적인' 활성화 결과를 확인하는 방법이라고 할 수 있습니다. Class Activation Map을 제안한 저자<small>(Bolei Zhou et al.)</small>의 논문에 수록된 아래 그림을 통해, 이러한 컨셉을 한 눈에 이해할 수 있습니다.

{% include image.html name=page.name file="class-activation-mapping.png" description="예측 클래스에 대해 얻어진 Class Activation Map 예시 <small>(Bolei Zhou et al.)</small>" class="full-image" %}

Saliency Map이 입력 이미지 상에서 Attribution을 수행하여 다소 산개된 점 형태의 결과물을 도출한다면, Class Activation Map은 컨볼루션 layer 상에서 Attribution을 수행하기 때문에 상대적으로 부드러운 Attribution 결과를 보여준다는 특징이 있습니다. 아래 그림을 통해 이러한 특징을 엿볼 수 있습니다.

{% include image.html name=page.name file="class-activation-map-examples.png" description="컨볼루션 신경망의 예측 결과에 대한 Class Activation Map 도출 결과 예시 <small>(Bolei Zhou et al.)</small>" class="large-image" %}


## Dataset Visualization

Attribution이 단일 입력 이미지에 대한 컨볼루션 신경망의 예측 결과에 대한 설명을 제공한다면, '**Dataset Visualization**(데이터셋 시각화)'은 데이터셋 상에 포함된 전체 이미지들에 대한 컨볼루션 신경망의 예측 결과의 일반적 경향성에 대한 설명을 제공합니다. 예를 들어, 아래 그림과 같이 하나의 컨볼루션 layer를 관찰 대상으로 고정해 놓고, 데이터셋 상의 이미지들을 하나씩 입력하여 이들 각각에 대한 feature map들을 산출한 뒤, 여기에 *Dimensionality Reduction*(차원 축소) 방법을 적용하여 2차원(2D) 또는 3차원(3D) feature space(요인 공간) 상의 점으로 도시할 수 있습니다.

{% include image.html name=page.name file="dataset-visualization-concept.png" description="컨볼루션 layer 상의 feature maps의 2D feature space로의 Dataset Visualization 예시" class="full-image" %}

Dataset Visualization을 위한 대표적인 Dimensionality Reduction 방법으로는 <a href="https://en.wikipedia.org/wiki/Principal_component_analysis" target="_blank">PCA(principal component analysis)</a>, <a href="http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf" target="_blank">t-SNE(t-Distributed Stochastic Neighbor Embedding)</a>, <a href="https://arxiv.org/pdf/1802.03426" target="_blank">UMAP</a> 등이 있습니다. 이들은 컨볼루션 신경망의 feature maps 뿐만 아니라, 다른 high-dimensional(고차원) 데이터의 시각화를 위해서도 범용적으로 적용할 수 있는 Dimensionality Reduction 방법에 해당합니다. 예를 들어 <a href="{{ site.url }}{% post_url 2017-11-29-image-recognition-overview-1 %}#인간의-인식-성능을-좇기-위한-도전" target="_blank">MNIST 데이터셋</a>으로 학습한 컨볼루션 신경망에 대하여 t-SNE를 적용할 경우, 아래 그림과 같은 형태의 Dataset Visualization 결과를 도출할 수 있습니다.

{% include image.html name=page.name file="tsne-visualization-example.png" description="MNIST 데이터셋에 대한 컨볼루션 신경망의 t-SNE Visualization (2D) 결과 예시 <small>(Laurens van der Maaten and Geoffrey Hinton)</small>" class="large-image" %}

혹은, <a href="http://www.image-net.org/challenges/LSVRC/2014/" target="_blank">ILSVRC</a>와 같은 데이터셋 상의 자연계 이미지에 대해서도 동일한 방법으로 t-SNE를 적용하고, 이를 통해 얻어진 2D feature space 상의 공간적 위치에 의거하여, 이미지들을 균일한 간격으로 배치한 아래와 같은 Dataset Visualization 결과도 도출할 수 있습니다.

{% include image.html name=page.name file="cnn-embed-full-1k.jpg" description="ILSVRC 데이터셋에 대한 컨볼루션 신경망의 t-SNE Visualization (2D) 결과를 원본 이미지로 표현한 예시 <small>(Andrej Karpathy)</small>" class="large-image" %}

MNIST 데이터셋과 ILSVRC 데이터셋에 대한 Dataset Visualization 결과 모두에서, 대체로 동일하거나 유사한 클래스 이미지들이 공간적으로 서로 모여 군집(cluster)을 이루고 있는 경향을 확인할 수 있습니다. 이와 같이 Dataset Visualization을 통해, 데이터셋 상에 포함된 전체 이미지들에 대한 컨볼루션 신경망의 예측 결과의 전반적인 경향성 및 각 예측 결과들 간의 거리 관계 등을 한 눈에 확인할 수 있습니다.


## 결론

지금까지 이미지 인식 문제에 적용된 딥러닝 모델, 즉 컨볼루션 신경망에 대한 대표적인 IML 방법론들을 확인해 보았습니다. 컨볼루션 신경망의 예측 결과에 대한 해석을 위한 가장 단순하고 직관적인 방법으로 feature map을 직접 이미지 형태로 시각화하는 Activation Visualization이 있는데, 늘 한 번에 많은 수의 feature map들을 동시에 관찰하면서 각각이 커버하는 시각적 특징이 무엇인지 추정해야 한다는 단점이 존재함을 확인하였습니다. 한편 학습된 컨볼루션 신경망의 weights 자체를 시각화하는 Weight Visualization의 경우, 예시 이미지들을 입력해 줄 필요가 없다는 장점이 있으나 그 결과에 대한 해석이 다소 난해하다는 문제가 있었습니다.

Activation Maximization은 컨볼루션 신경망의 다양한 중간 출력값들이 커버하는 시각적 특징을 좀 더 효과적으로 확인할 수 있도록 하는 방법입니다. 컨볼루션 신경망 상의 특정 타겟 출력값을 최대로 활성화하는 입력 이미지들을 현재 가지고 있는 데이터셋 상에서 '탐색'하여 Maximally Activating Images를 얻거나, 혹은 gradient ascent에 기반한 optimization을 통해 이를 직접 '생성'하는 Maximization by Optimization을 시도할 수 있음을 확인하였습니다. 관심의 대상이 되는 타겟 출력값으로는 neuron, feature map(=channel), layer 혹은 prediction layer의 logit 등이 될 수 있으며, 이를 어떻게 설정하느냐에 따라 Activation Maximization 수행 결과가 크게 달라짐을 확인할 수 있었습니다.

반면 컨볼루션 신경망의 중간 출력값보다는 예측 결과 자체에 집중하여 여기에 대한 '설명'을 제공하기 위한 방법으로 Attribution이 있습니다. Saliency Map은 예측 클래스 logit의 입력 이미지에 대한 gradient를 계산하여 생성해 낸 Attribution 수단으로, 이를 관찰함으로써 컨볼루션 신경망의 특정 예측 결과가 이미지 상의 어느 부분에 기인하였는지 가시적으로 확인할 수 있습니다. 한편 Class Activation Map은 컨볼루션 layer의 feature map들에 대한 가중합을 계산하는 방식을 통해 Attribution 결과를 생성해 낸 결과물로, 좀 더 부드러운 Attribution 결과를 보여준다는 특징이 있습니다.

만일 어느 데이터셋 상의 전체 이미지들에 대한 컨볼루션 신경망의 예측 결과들을 조망하고 이들의 일반적 경향성을 확인하고 싶은 경우, PCA, t-SNE, UMAP과 같은 Dimensionality Reduction 방법을 적용하여 Dataset Visualization을 시도해볼 수 있습니다. 이를 통해 컨볼루션 신경망이 주어진 데이터셋에 대하여 좀 더 거시적인 맥락에서 어떻게 동작하도록 학습되었는지 간접적으로 파악할 수 있음을 확인하였습니다.


## References

- Jason Yosinski et al. Understanding neural networks through deep visualization.
  - <a href="https://arxiv.org/pdf/1506.06579" target="_blank">Yosinski, Jason, et al. "Understanding neural networks through deep visualization." arXiv preprint arXiv:1506.06579 (2015).</a>
- Jost Tobias Springenberg et al. Striving for Simplicity: The All Convolutional Net.
  - <a href="https://arxiv.org/pdf/1412.6806.pdf%20(http://arxiv.org/pdf/1412.6806.pdf)" target="_blank">Springenberg, Jost Tobias, et al. "Striving for simplicity: The all convolutional net." arXiv preprint arXiv:1412.6806 (2014).</a>
- Karen Simonyan et al. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.
  - <a href="https://arxiv.org/pdf/1312.6034.pdf" target="_blank">Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013).</a>
- Anh Nguyen et al. Multifaceted feature visualization: Uncovering the different types of features learned by each neuron in deep neural networks.
  - <a href="https://arxiv.org/pdf/1602.03616" target="_blank">Nguyen, Anh, Jason Yosinski, and Jeff Clune. "Multifaceted feature visualization: Uncovering the different types of features learned by each neuron in deep neural networks." arXiv preprint arXiv:1602.03616 (2016).</a>
- Bolei Zhou et al. Learning Deep Features for Discriminative Localization.
  - <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf" target="_blank">Zhou, Bolei, et al. "Learning deep features for discriminative localization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.</a>
- David Bau et al. Network Dissection: Quantifying Interpretability of Deep Visual Representations.
  - <a href="http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf" target="_blank">Bau, David, et al. "Network dissection: Quantifying interpretability of deep visual representations." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.</a>
- Google Brain 팀에서 포스팅한 블로그 글 (1), (2)
  - <a href="https://distill.pub/2017/feature-visualization" target="_blank">Christopher Olah, et al., "Feature Visualization", Distill, 2017.</a>
  - <a href="https://distill.pub/2018/building-blocks" target="_blank">Christopher Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.</a>
- 컨볼루션 신경망의 각 layer에서의 activation visualization 결과 예시, ILSVRC 데이터셋에 대한 컨볼루션 신경망의 t-SNE Visualization (2D) 결과를 원본 이미지로 표현한 예시
  - <a href="http://cs231n.github.io/convolutional-networks" target="_blank">Andrej Karpathy, "Convolutional Neural Networks for Visual Recognition." Stanford University, http://cs231n.github.io/convolutional-networks. Accessed 5 October 2019.</a>
- 타겟 출력값을 서로 다른 layer로 설정했을 시의 서로 다른 Maximization by Optimization(DeepDream) 수행 결과 예시
  - <a href="https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/r1/deepdream.ipynb" target="_blank">DeepDreaming with TensorFlow</a>
- PCA(principal component analysis)
  - <a href="https://en.wikipedia.org/wiki/Principal_component_analysis" target="_blank">Wikipedia contributors. "Principal component analysis." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 9 Oct. 2019. Web. 22 Oct. 2019. </a>
- t-SNE(t-Distributed Stochastic Neighbor Embedding)
  - <a href="http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf" target="_blank">Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.Nov (2008): 2579-2605.</a>
- UMAP(uniform manifold approximation and projection)
  - <a href="https://arxiv.org/pdf/1802.03426" target="_blank">McInnes, Leland, John Healy, and James Melville. "Umap: Uniform manifold approximation and projection for dimension reduction." arXiv preprint arXiv:1802.03426 (2018).</a>
