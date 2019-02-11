---
layout: post
title: "이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로 (1)"
date: 2017-11-29 09:00:00 +0900
author: kilho_kim
categories: [Introduction]
tags: [pascal voc, classification, detection, segmentation]
comments: true
name: image-recognition-overview-1
redirect_from: "/computer-vision/2017/11/29/image-recognition-overview-1.html"
---

지난 번 글까지 해서 수아랩의 핵심 기술들 중 하나인 '딥러닝'에 대해 알아보았습니다. 오늘날 딥러닝 기술이 적용되고 있는 분야는 이미지 인식, 음성 인식, 자연어 처리 등 여러 가지가 있습니다. 오늘은 이러한 적용 분야들 중, 딥러닝의 위력을 가장 드라마틱하게 보여주고 있다고 할 수 있는 '이미지 인식' 분야에서 다루는 문제들을 정의하고, 이들의 주요 목표가 무엇인지, 모델의 예측 결과를 어떤 척도로 평가하는지 등에 대하여 살펴보고자 합니다. 우선 이미지 인식 분야에 대한 이해를 완벽하게 가져간 후에, 여기에 적용되는 딥러닝 기술에 대하여 추후에 자세히 살펴보도록 하겠습니다.

- 본문의 플롯을 위해 작성한 <a href="https://github.com/sualab/sualab.github.io/blob/master/assets/notebooks/image-recognition-overview.ipynb" target="_blank">Python 코드</a>를 부록으로 함께 첨부하였습니다. 

## 서론

**이미지 인식(image recognition)** 문제에서는, 기계로 하여금 주어진 이미지 상에 포함되어 있는 대상이 *무엇인지*, 또한 *어느 위치에 있는지* 등을 파악하도록 하는 것을 주된 목표로 합니다. 예를 들어, 수아랩 기술 블로그를 오랫동안 보아 오셨다면 너무나도 친숙할 만한, 아래와 같은 이미지가 주어졌다고 합시다.

{% include image.html name=page.name file="tree-image.png" description="인간이 받아들이는 나무 이미지" class="large-image" %}

5살 남짓의 어린 아이조차도, 위 이미지를 관찰한 순간 그 안에 '나무'라는 대상이 포함되어 있다는 것을 불과 0.1초 내로 *빠르고 정확하게* 인식할 수 있습니다. 비단 나무뿐만 아니라, 어린 아이는 그 주변에 존재하는 다양한 대상들에 대해서도 큰 무리 없이 유사한 속도와 성능(?)으로 인식할 것이라고 쉽게 예상할 수 있습니다.

그러나 오늘날 과학 기술이 꽃을 피운 21세기에 접어들었음에도 불구하고, 이렇게 어린 아이조차도 쉽게 할 수 있는 이미지 인식이, 기계에게는 여전히 매우 어려운 일로 받아들여지고 있습니다. 지난 <a href="http://research.sualab.com/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a> 글에서도 언급하였듯이, 기계는 이미지를 **픽셀(pixel)** 단위의 수치화된 형태로 받아들이며, 일반적으로 인간이 보고 이해할 수 있을만큼 큰 이미지는 매우 많은 수의 픽셀들로 구성되어 있습니다. 

{% include image.html name=page.name file="tree-image-pixels.svg" description="기계가 받아들이는 나무 이미지: 수많은 픽셀을 통한 표현*<br><small>(*주의: 격자 안의 하나의 정사각형의 크기는 실제 1픽셀보다는 크며, 설명을 돕기 위해 과장하였습니다.)</small>" class="large-image" %}

> 위 나무 이미지는, 실제로는 756x409(=309,204)개의 픽셀로 이루어져 있습니다.

위와 같은 이미지를 보고 '나무'라는 추상적인 개념을 뽑아내는 작업에 있어, 인간의 경우 (아직 완전히 밝혀지지 않은 모종의 매커니즘에 의해) '선택적 주의 집중(selective attention)' 및 '문맥(context)'에 기반한 '종합적 이해' 등의 과정을 거치며, 이 작업을 *직관적으로* 빠른 속도로 정확하게 수행할 수 있습니다. 반면, 기계는 '선택적 주의 집중' 능력이 없기 때문에 픽셀의 값을 빠짐없이 하나하나 다 살펴봐야 하므로 일단 이 과정에서 속도가 느려질 수밖에 없으며, 이렇게 읽어들인 픽셀로부터 어떻게 '문맥' 정보를 추출하고, 또 이들을 어떻게 '종합하고 이해'하는 것이 최적인지도 알지 못하므로 그 성능 또한 인간에 한참 뒤떨어질 수밖에 없습니다.

### 인간의 인식 성능을 좇기 위한 도전

이러한 상황에서, 기계의 이미지 인식 속도와 성능을 인간의 수준으로 끌어올리기 위한 가장 효과적인 방법은 '인간이 이미지를 인식하는 매커니즘을 밝혀내고, 이를 기계로 하여금 모방하도록 해 보자'는 것이라고 생각할 수 있습니다. 실제로, 이는 뇌 과학(brain science) 분야에서 주로 다루어지는 연구 주제입니다. 이를 위해서는 인간의 지능을 구성하는 지식 표현, 학습, 추론, 창작 등에 해당하는 인공지능 문제들이 모두 풀려야 가능할 것으로 보이니, 이 방향으로 가기에는 아직 갈 길이 한참 먼 것이 현실입니다.

이미지 인식 연구 초창기에 뇌 과학의 연구 성과를 마냥 기다릴 수만은 없었던 공학자들은, 인간의 인식 메커니즘을 그대로 모방하려는 시도 대신, 기존의 이미지 인식 문제의 범위를 좁혀서 좀 더 특수한 목적을 지니는 쉬운 형태의 문제로 치환하고 이들을 수학적 기법을 통해 해결하는 방법을 고안해 왔습니다. 예를 들어, 인간의 '선택적 주의 집중' 및 '문맥 파악' 능력에는 못 미치지만, 어떤 특수한 문제 해결에 효과적인 **요인(feature)**을 정의하여 사용하고, 이들을 '종합하고 이해'하도록 하기 위해 **러닝 모델(learning model)**과 **러닝 알고리즘(learning algorithm)**을 사용하여 이를 머신러닝 차원으로 해결하고자 하였습니다. 특수한 이미지 인식 문제로는 *얼굴 인식(face recognition)*, *필적 인식(handwriting recognition)* 등이 대표적입니다.

{% include image.html name=page.name file="face-recognition-examples.png" description="특수한 이미지 인식 문제 예시: 얼굴 인식(FERET database)" class="medium-image" %}

{% include image.html name=page.name file="mnist-handwriting-examples.png" description="특수한 이미지 인식 문제 예시: 필적 인식(MNIST database)" class="small-image" %}

초창기의 이러한 시도들을 통해 자신감을 얻은 공학자들은, 좀 더 과감한 도전을 하기 시작하였습니다. 인간이 일상 속에서 접할 수 있는 몇 가지 주요한 사물들을 인식하기 위한 시도를 시작한 것입니다. 이는, 기계의 이미지 인식 성능의 벤치마크(benchmark)로 삼을 수 있는 다양한 데이터셋이 등장한 데에서부터 출발하였습니다. 예를 들어, *CIFAR-10 dataset*은 일반적인 이미지 인식을 위한 가장 대표적인 벤치마크용 데이터셋으로, 32x32 크기의 작은 컬러 이미지 상에 10가지 사물 중 어떤 것이 포함되어 있는지를 단순 분류하는 문제를 제시하기 위해 만들어졌습니다.

{% include image.html name=page.name file="cifar10-examples.png" description="일반적인 이미지 인식 데이터셋 예시: CIFAR-10" class="large-image" %}

### 이미지 인식 문제의 정립: Classification, Detection, Segmentation

연구실 차원에서의 이런 올망졸망한(?) 벤치마크 데이터셋에서 출발하여, 그 후에는 1만 장 이상의 거대한 스케일의 이미지 데이터셋에 대하여 인식 성능을 겨루는 대회가 본격적으로 등장하였습니다. 초창기의 이미지 인식 대회 중 가장 대표적인 것이 *PASCAL VOC Challenge*입니다. 이 대회를 기점으로, 이미지 인식에서 다루는 문제들이 어느 정도 정형화되었다고 할 수 있습니다.

{% include image.html name=page.name file="classification-detection-segmentation.png" description="PASCAL VOC Challenge 문제: Classification, Detection, Segmentation" class="large-image" %}

PASCAL VOC Challenge를 기준으로 볼 때, 이미지 인식 분야에서 다루는 주요 문제를 크게 3가지로 정리할 수 있습니다. **Classification**, **Detection**, **Segmentation**이 바로 그것입니다. 지금부터 이들 각각의 문제가 무엇인지 정의하고, 각 문제와 관련된 주요한 이슈는 무엇인지, 어떤 기준으로 예측 성능을 평가하는지 순으로 이야기해 보도록 하겠습니다.

## Classification

### 문제 정의

Classification 문제에서는, *주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 포함되어 있는지 여부를 분류하는 모델을 만드는 것*을 주요 목표로 합니다. 여기에서 **클래스(class)**란, 분류 대상이 되는 카테고리 하나하나를 지칭합니다. 

본격적인 Classification을 수행하기 전에, 반드시 관심의 대상이 되는 클래스들을 미리 정해놓고 작업을 시작해야 합니다. 예를 들어, PASCAL VOC Challenge에서는 총 20가지 클래스를 상정하고, 이에 대한 classification을 수행하도록 하였습니다.

{% include image.html name=page.name file="pascal-voc-classes.png" description="PASCAL VOC Challenge에서 다루는 20가지 클래스<br><small>(좌측 절반 10개: 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',<br> 우측 절반 10개: 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'TV/monitor')</small>" class="full-image" %}

PASCAL VOC Challenge를 비롯한 대부분의 이미지 인식 대회의 Classification 문제에서는, 주어진 이미지 안에 특정 클래스의 사물이 존재할 '가능성' 내지는 '믿음'을 나타내는 **신뢰도 점수(confidence score)**를 제출하도록 요구합니다. 즉, '주어진 이미지 안에 클래스 X의 사물이 있다'는 식의 단정적인 결론 대신, '주어진 이미지 안에 클래스 X의 사물이 존재할 가능성이 $$s_X$$, 클래스 Y의 사물이 존재할 가능성이 $$s_Y$$, 클래스 Z의 사물이 존재할 가능성이 $$s_Z$$, ...' 식의 결과물을 제출하도록 요구하고, 이를 통해 추후 정답 여부 확인 시 해당 결과물에 대한 사후적인 해석의 여지를 두게 되는 것입니다.

{% include image.html name=page.name file="classification-model.svg" description="Classification 문제<br><small>(예시 이미지: VOC2009 데이터셋 - 2009_001984.jpg)</small>" class="full-image" %}

#### 신뢰도 점수에 대한 해석 방법

Classification 문제에서 분류의 대상이 되는 이미지에는 반드시 하나의 사물만이 포함되어 있거나, 또는 복수 개의 서로 다른 사물들이 포함되어 있을 수도 있습니다. 둘 중 어느 경우를 전제하느냐에 따라, 신뢰도 점수에 대한 최종적인 해석 방법이 달라집니다. 

먼저 *모든 이미지가 반드시 하나의 사물만을 포함하도록* 전제되어 있는 경우를 생각해 봅시다. 이를 편의 상 '*단일 사물 분류*' 문제라고 지칭하도록 하겠습니다. 이 경우, 전체 클래스에 대한 신뢰도 점수 중 가장 큰 신뢰도 점수를 갖는 클래스를 선정하여, '주어진 이미지 안에 해당 클래스가 포함되어 있을 것이다'고 결론지을 수 있습니다. 예를 들어, 아래와 같이 '고양이'를 담고 있는 이미지가 주어졌을 때, 전체 20가지 클래스에 대한 신뢰도 점수들을 비교하여 그 중 가장 큰 신뢰도 점수를 지니는 'cat' 클래스를 선정하여 제시할 수 있습니다. 

{% include image.html name=page.name file="single-object-classification-confidence-scores.svg" description="단일 사물 분류 문제에서의 신뢰도 점수 해석<br><small>(예시 이미지: VOC2008 데이터셋 - 2008_005977.jpg)</small>" class="large-image" %}

단일 사물 분류를 요구하는 데이터셋으로는 앞서 언급했던 MNIST, CIFAR-10 등이 있으며, 이들은 상대적으로 쉬운 문제로 취급됩니다. 

반면, 이번에는 *이미지 상에 복수 개의 사물들이 포함되어 있을 수 있도록* 전제되어 있는 경우입니다. 이를 '*복수 사물 분류*' 문제라고 지칭하도록 하겠습니다. 이 경우, 단순히 위와 같이 가장 큰 신뢰도 점수를 갖는 클래스 하나만을 선정하여 제시하는 것은 그다지 합리적인 결론이 아닐 것입니다. 

이러한 문제 상황에서는 이미지 인식 대회마다 결론을 도출하는 방식이 조금씩 다르나, PASCAL VOC Challenge의 경우에는 각 클래스마다 **문턱값(threshold)**을 미리 설정해 놓고, 주어진 이미지의 *각 클래스 별 신뢰도 점수가 문턱값보다 **큰** 경우에 한하여 '주어진 이미지 안에 해당 클래스가 포함되어 있을 것이다'고 결론*짓도록 합니다. 예를 들어, 아래와 같이 '소'와 '사람'을 동시에 담고 있는 이미지가 주어졌을 때, 20가지 클래스 각각의 신뢰도 점수들을 조사하여, 이들 중 사전에 정한 문턱값보다 큰 신뢰도 점수를 지니는 'cow'와 'person' 클래스를 선정하여 제시할 수 있습니다.

{% include image.html name=page.name file="multiple-objects-classification-confidence-scores.svg" description="복수 사물 분류 문제에서의 신뢰도 점수 해석<br><small>(예시 이미지: VOC2010 데이터셋 - 2010_001692.jpg)</small>" class="large-image" %}

> 그렇다면, 각 클래스의 문턱값은 어떻게 결정해야 할까요? 이는 어느 평가 척도를 사용하여 평가할지의 문제와 같이 엮어 고민해야 하는 문제입니다.

복수 사물 분류 문제가 아무래도 현실 상황에 좀 더 부합한다고 할 수 있으며, 상대적으로 좀 더 어려운 문제로 취급됩니다. PASCAL VOC Challenge, ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 등 주요한 이미지 인식 대회에서 이를 채택하고 있습니다.

### 평가 척도

#### 정확도(accuracy)

어떤 모델의 Classification 성능을 평가하고자 할 때, 다양한 종류의 **평가 척도(evaluation measure)** 중 하나 혹은 여러 개를 선정하여 사용할 수 있습니다. 일반적으로 가장 쉽게 떠올릴 수 있는 척도로 **정확도(accuracy)**가 있습니다. Classification 문제에서의 정확도는 일반적으로, *테스트를 위해 주어진 전체 이미지 수 대비, 분류 모델이 올바르게 분류한 이미지 수*로 정의합니다. 

\begin{equation}
\text{정확도} = \frac{\text{올바르게 분류한 이미지 수}} {\text{전체 이미지 수}}
\end{equation}

단일 사물 분류 문제에서는,  위에서 정의된 정확도를 평가 척도로 즉각 사용하여도 크게 문제가 없습니다. 예를 들어, 아래와 같이 전체 테스트용 이미지가 10개 있었다고 할 때, 분류 모델이 이들 중 7개를 올바르게 예측했다면, 정확도는 $$7 / 10 = 0.7$$($$70\%$$)이 됩니다.

{% include image.html name=page.name file="accuracy-example.svg" description="단일 사물 분류 문제에서의 정확도 계산 예시" class="large-image" %}

#### 정밀도(precision)와 재현율(recall)

그러나, 복수 사물 분류 문제에서는, 위의 정확도를 그대로 사용하기 곤란해지는 상황이 발생합니다. 이 때문에, 정확도 대신 **정밀도(precision)** 및 **재현율(recall)** 등의 평가 척도를 사용합니다. 정밀도와 재현율은 하나의 클래스에 대하여 (다른 클래스와는 독립적으로) 매겨지는 평가 척도입니다.

Classification 문제에서의 어느 특정 클래스 $$c$$의 정밀도는, *분류 모델이 $$c$$일 것으로 예측한 이미지 수 대비, 분류 모델이 올바르게 분류한 클래스 $$c$$ 이미지 수*로 정의합니다. 한편, 클래스 $$c$$의 재현율은, *전체 클래스 $$c$$ 이미지 수 대비, 분류 모델이 올바르게 분류한 클래스 $$c$$ 이미지 수*로 정의합니다.

\begin{equation}
\text{클래스 c의 정밀도} = \frac{\text{올바르게 분류한 클래스 c 이미지 수}} {\text{클래스 c일 것으로 예측한 이미지 수}}
\end{equation}

\begin{equation}
\text{클래스 c의 재현율} = \frac{\text{올바르게 분류한 클래스 c 이미지 수}} {\text{전체 클래스 c 이미지 수}}
\end{equation}

각 클래스에 대한 정밀도 및 재현율을 계산한 뒤, 이들 전체의 대표값(representative value)을 취하고, 이를 최종적인 평가 척도로 삼을 수 있습니다. 구체적으로, 전체 $$C$$개 클래스에 대한 평균 정밀도 및 평균 재현율을 계산하고자 한다면, 아래와 같은 공식을 사용할 수 있습니다(이 때, '클래스 $$c$$ 이미지'란 클래스 $$c$$에 해당하는 사물을 포함하고 있는 이미지를 지칭합니다).

\begin{equation}
\text{평균 정밀도} = \frac{1}{C} \sum_{c=1}^{C} \text{(클래스 c의 정밀도)}
\end{equation}

\begin{equation}
\text{평균 재현율} = \frac{1}{C} \sum_{c=1}^{C} \text{(클래스 c의 재현율)}
\end{equation}

> 총으로 사냥을 하는 것에 비유하자면, 일단 발사한 탄환 하나마다 사냥감 하나씩을 반드시 놓치지 않고 맞추도록 하고자 한다면, 정밀도를 높이는 방향으로 전략을 짜야 합니다. 반면, '헛방'이 많이 나도 좋으니 어떻게든 자기 주변에 있는 모든 사냥감을 맞추는 것이 목표라면, 재현율을 높이는 방향으로 전략을 짜야 합니다.

평균 정밀도를 계산하는 구체적인 과정을 보면, 아래 그림과 같이 원본 테스트 이미지들을 모델이 예측한 클래스를 기준으로 나눈 후, 각각에 대하여 정밀도를 따로 계산한 뒤, 이렇게 얻어진 클래스 별 정밀도의 평균을 계산합니다.

{% include image.html name=page.name file="precision-per-class-example.svg" description="복수 사물 분류 문제에서의 클래스 별 정밀도 계산 예시<br><small>(그림에 제시된 3개의 클래스에 대한 전체 평균 정밀도는 $$(0.4+0.6+0.4)/3 = 0.47(47\%)$$)</small>" class="large-image" %}

다음으로 평균 재현율을 계산하는 구체적인 과정을 보면, 아래 그림과 같이 원본 테스트 이미지들을 실제 클래스를 기준으로 나눈 후, 각 클래스에 대하여 재현율을 따로 계산한 뒤, 이렇게 얻어진 클래스 별 재현율의 평균을 계산합니다.

{% include image.html name=page.name file="recall-per-class-example.svg" description="복수 사물 분류 문제에서의 클래스 별 재현율 계산 예시<br><small>(그림에 제시된 3개의 클래스에 대한 전체 평균 재현율은 $$(0.6+1.0+0.8)/3 = 0.8(80\%)$$)</small>" class="large-image" %}

#### 신뢰도 점수의 문턱값에 따른 평가 척도 수치의 변화 가능성

복수 사물 분류 문제의 경우, 각 클래스 별로 신뢰도 점수에 대한 문턱값을 어떻게 결정해야 하는지에 대한 이슈가 여전히 남아 있습니다. 이해를 돕기 위해, 'car' 클래스에 대한 분류 모델의 신뢰도 점수가 주어졌을 때, 특정 문턱값에 따라 결론을 내리는 상황을 살펴보도록 하겠습니다. 이 때, 편의 상 주어진 이미지를 'car' 클래스로 예측하지 *않은* 경우를 not 'car' 클래스라고 지칭하도록 하겠습니다.

먼저, (1) *'car' 클래스의 문턱값을 높게 잡을수록, 분류 모델이 'car' 클래스로 예측하게 되는 이미지의 개수가 감소*합니다. 이렇게 되면, 신뢰도 점수가 확실하게 높은 이미지에 대해서만 'car' 클래스로 예측하게 되므로 *정밀도가 상승*하나, 반대로 실제 존재하는 많은 수의 'car' 이미지들을 놓치게 되므로 *재현율은 하락*합니다.

반면에 (2) *'car' 클래스의 문턱값을 낮게 잡을수록, 분류 모델이 'car' 클래스로 예측하게 되는 이미지의 개수가 증가*합니다. 이렇게 되면, 신뢰도 점수가 낮은 이미지들까지 공격적으로 'car' 클래스로 예측하게 되므로 *재현율이 상승*하나, 반대로 많은 수의 not 'car' 이미지들마저 모조리 'car' 클래스로 예측하게 되므로 *정밀도는 하락*합니다.

(1)과 (2)의 상황에서 확인할 수 있듯이, *정밀도와 재현율 간에는 서로 약한 trade-off 관계가 존재*합니다. 좀 더 구체적으로 'car' 클래스에 대하여, 테스트 이미지들에 대한 분류 모델의 신뢰도 점수가 계산된 상황에서, 문턱값의 변화에 따라 모델의 예측 결과 및 실제 정답 여부를 아래 그림과 같이 나타냈습니다. 

{% include image.html name=page.name file="threshold-to-classification-results.svg" description="'car' 클래스의 문턱값에 따른, 정밀도 및 재현율 결과 변화 표<br><small>(정밀도 n/a의 경우, 클래스 $$c$$로 예측한 이미지 수가 0개이므로 계산이 불가함을 나타냄)</small>" class="full-image" %}

위 그림에서는 편의 상 문턱값을 $$1.0$$ 간격으로 조정하면서 정밀도 및 재현율을 측정한 것인데, 문턱값의 조정 간격을 더 짧게 하고 정밀도와 재현율을 측정하면 아래와 같은 형태의 플롯을 얻을 수 있습니다.

{% include image.html name=page.name file="precision-recall-to-threshold-plot.svg" description="'car' 클래스의 문턱값에 따른, 정밀도 및 재현율 결과 변화 플롯" class="medium-image" %}

위 플롯에서, 정밀도 혹은 재현율이 변화하는 지점만을 포착하여, 이들 지점에서의 $$(재현율, 정밀도)$$를 아래 그림과 같이 2차원 평면 상에 나타내는 것이 더 일반적인 표현 방법에 해당합니다. 이를 **정밀도-재현율 곡선(precision-recall curve)**이라고 부릅니다.

{% include image.html name=page.name file="precision-recall-curve-plot.svg" description="'car' 클래스의 문턱값에 따른, 정밀도-재현율 곡선 플롯" class="medium-image" %}

위의 사례에서는, 'car' 클래스의 문턱값을 약 $$2.3$$ 정도로 설정했을 때, 정밀도 및 재현율 모두 $$0.8$$로 적당히 높은 수치를 기록했습니다. 아마도 여러분들께서는 위와 같이 문턱값을 조정하면서 테스트 이미지들에 대한 채점 결과를 관찰하여, 높은 수치의 정밀도 혹은 재현율을 발휘하는 문턱값을 결정하면 될 것 같다는 충동이 들 것입니다.

그런데, 사실 이런 방식으로 최적의 문턱값을 결정하여 최종 성능을 뽑아내면, 그 결과는 현재 가지고 있는 테스트 이미지들에 한해서만 지나치게 '낙관적인(optimistic)' 결과가 되어 버립니다. 즉, 새로운 테스트 이미지가 들어오는 상황에서 발휘할 수 있는 '일반적인 성능'이라고 담보하기 어려워지는 것입니다. 

> 이는 마치 시험 시작 직전에 시험 출제 문제를 1분 정도 슬쩍 컨닝한 뒤 시험을 보는 것과 같은 행동입니다. 

이러한 맹점을 보완하고자 대부분의 이미지 인식 대회에서는, 문턱값을 특정 값으로 한정시킨 상황에서의 성능 척도만을 보는 것이 아니라, 문턱값이 존재할 수 있는 전체 범위 내에서의 정밀도 및 재현율들을 계산하고, 이들의 대푯값을 계산하는 방법을 채택하고 있습니다. 

예를 들어, PASCAL VOC Challenge에서는 **평균 정밀도(average precision)**라는 평가 척도를 사용합니다. 평균 정밀도는, 각 문턱값에서 얻어지는 정밀도를, (이전 문턱값에서와 비교한)재현율의 증가량으로 곱한 것들의 총합으로 정의되며, 단순하게 생각하면 *정밀도-재현율 곡선과 재현율 축 사이의 넓이*에 해당합니다.

\begin{equation}
\text{평균 정밀도} = \sum_t (R_t - R_{t-1}) \cdot P_t
\end{equation}

### 의의

Classification 문제는, 이어질 Detection 및 Segmentation 문제를 향한 출발점이라고 할 수 있습니다. Detection 및 Segmentation 문제 해결을 위해서는 특정 클래스에 해당하는 사물이 이미지 상의 어느 곳에 위치하는지에 대한 정보를 파악해야 하는데, 이를 위해서는 우선 그러한 사물이 이미지 상에 존재하는지 여부가 반드시 먼저 파악되어야 하기 때문입니다. 

이러한 경향 때문에, Classification 문제에서 우수한 성능을 발휘했던 모델을 Detection 또는 Segmentation을 위한 구조로 변형하여 사용할 경우, 그 역시 상대적으로 우수한 성능을 발휘하는 경향이 있습니다.

[(다음 포스팅 보기)]({{ page.url }}/../image-recognition-overview-2.html)
