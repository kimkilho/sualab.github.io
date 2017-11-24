---
layout: post
title: "이미지 인식 문제와 딥러닝"
date: 2017-11-05 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, machine-vision]
tags: [machine-learning, data-science, machine-vision]
comments: true
name: image-recognition-overview
---

지난 번 글까지 해서 수아랩의 핵심 기술들 중 하나인 '딥러닝'에 대해 알아보았습니다. 오늘날 딥러닝 기술이 적용되고 있는 분야는 이미지 인식, 음성 인식, 자연어 처리 등 여러 가지가 있습니다. 오늘은 이러한 적용 분야들 중, 딥러닝의 위력을 가장 드라마틱하게 보여주고 있다고 할 수 있는 '이미지 인식' 분야에서 다루는 문제들을 언급하고, 오늘날 딥러닝 기술을 활용하여 이들 문제에 어떻게 접근하고 있는지에 대하여 살펴보고자 합니다. 

## 서론

**이미지 인식(image recognition)** 문제에서는, 기계로 하여금 주어진 이미지 상에 포함되어 있는 대상이 *무엇인지*, 또한 *어느 위치에 있는지* 등을 파악하도록 하는 것을 주된 목표로 합니다. 예를 들어, 수아랩 기술 블로그를 오랫동안 보아 오셨다면 너무나도 친숙할 만한, 아래와 같은 이미지가 주어졌다고 합시다.

{% include image.html name="image-recognition-overview" file="tree-image.png" description="인간이 받아들이는 나무 이미지" class="large-image" %}

5살 남짓의 어린 아이조차도, 위 이미지를 관찰한 순간 그 안에 '나무'라는 대상이 포함되어 있다는 것을 불과 0.1초 내로 *빠르고 정확하게* 인식할 수 있습니다. 비단 나무뿐만 아니라, 어린 아이는 그 주변에 존재하는 다양한 대상들에 대해서도 큰 무리 없이 유사한 속도와 성능(?)으로 인식할 것이라고 쉽게 예상할 수 있습니다.

그러나 오늘날 과학 기술이 꽃을 피운 21세기에 접어들었음에도 불구하고, 이렇게 어린 아이조차도 쉽게 할 수 있는 이미지 인식이, 기계에게는 여전히 매우 어려운 일로 받아들여지고 있습니다. 지난 <a href="http://research.sualab.com/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a> 글에서도 언급하였듯이, 기계는 이미지를 **픽셀(pixel)** 단위의 수치화된 형태로 받아들이며, 일반적으로 인간이 보고 이해할 수 있을만큼 큰 이미지는 매우 많은 수의 픽셀들로 구성되어 있습니다. 

{% include image.html name="image-recognition-overview" file="tree-image-pixels.svg" description="기계가 받아들이는 나무 이미지: 수많은 픽셀을 통한 표현*<br><small>(*주의: 격자 안의 하나의 정사각형의 크기는 실제 1픽셀보다는 크며, 설명을 돕기 위해 과장하였습니다.)</small>" class="large-image" %}

> 위 나무 이미지는, 실제로는 756x409(=309,204)개의 픽셀로 이루어져 있습니다.

위와 같은 이미지를 보고 '나무'라는 추상적인 개념을 뽑아내는 작업에 있어, 인간의 경우 (아직 완전히 밝혀지지 않은 모종의 매커니즘에 의해) '선택적 주의 집중(selective attention)' 및 '문맥(context)'에 기반한 '종합적 이해' 등의 과정을 거치며, 이 작업을 *직관적으로* 빠른 속도로 정확하게 수행할 수 있습니다. 반면, 기계는 '선택적 주의 집중' 능력이 없기 때문에 픽셀의 값을 빠짐없이 하나하나 다 살펴봐야 하므로 일단 이 과정에서 속도가 느려질 수밖에 없으며, 이렇게 읽어들인 픽셀로부터 어떻게 '문맥' 정보를 추출하고, 또 이들을 어떻게 '종합하고 이해'하는 것이 최적인지도 알지 못하므로 그 성능 또한 인간에 한참 뒤떨어질 수밖에 없습니다.

### 인간의 인식 성능을 좇기 위한 도전

이러한 상황에서, 기계의 이미지 인식 속도와 성능을 인간의 수준으로 끌어올리기 위한 가장 효과적인 방법은 '인간이 이미지를 인식하는 매커니즘을 밝혀내고, 이를 기계로 하여금 모방하도록 해 보자'는 것이라고 생각할 수 있습니다. 실제로, 이는 뇌 과학(brain science) 분야에서 주로 다루어지는 연구 주제입니다. 이를 위해서는 인간의 지능을 구성하는 지식 표현, 학습, 추론, 창작 등에 해당하는 인공지능 문제들이 모두 풀려야 가능할 것으로 보이니, 이 방향으로 가기에는 아직 갈 길이 한참 먼 것이 현실입니다.

이미지 인식 연구 초창기에 뇌 과학의 연구 성과를 마냥 기다릴 수만은 없었던 공학자들은, 인간의 인식 메커니즘을 그대로 모방하려는 시도 대신, 기존의 이미지 인식 문제의 범위를 좁혀서 좀 더 특수한 목적을 지니는 쉬운 형태의 문제로 치환하고 이들을 수학적 기법을 통해 해결하는 방법을 고안해 왔습니다. 예를 들어, 인간의 '선택적 주의 집중' 및 '문맥 파악' 능력에는 못 미치지만, 어떤 특수한 문제 해결에 효과적인 **요인(feature)**을 정의하여 사용하고, 이들을 '종합하고 이해'하도록 하기 위해 **러닝 모델(learning model)**과 **러닝 알고리즘(learning algorithm)**을 사용하여 이를 머신러닝 차원으로 해결하고자 하였습니다. 특수한 이미지 인식 문제로는 *얼굴 인식(face recognition)*, *필적 인식(handwriting recognition)* 등이 대표적입니다.

{% include image.html name="image-recognition-overview" file="face-recognition-examples.png" description="특수한 이미지 인식 문제 예시: 얼굴 인식(FERET database)" class="medium-image" %}

{% include image.html name="image-recognition-overview" file="mnist-handwriting-examples.png" description="특수한 이미지 인식 문제 예시: 필적 인식(MNIST database)" class="small-image" %}

초창기의 이러한 시도들을 통해 자신감을 얻은 공학자들은, 좀 더 과감한 도전을 하기 시작하였습니다. 인간이 일상 속에서 접할 수 있는 몇 가지 주요한 사물들을 인식하기 위한 시도를 시작한 것입니다. 이는, 기계의 이미지 인식 성능의 벤치마크(benchmark)로 삼을 수 있는 다양한 데이터셋이 등장한 데에서부터 출발하였습니다. 예를 들어, *CIFAR-10 dataset*은 일반적인 이미지 인식을 위한 가장 대표적인 벤치마크용 데이터셋으로, 32x32 크기의 작은 컬러 이미지 상에 10가지 사물 중 어떤 것이 포함되어 있는지를 단순 분류하는 문제를 제시하기 위해 만들어졌습니다.

{% include image.html name="image-recognition-overview" file="cifar10-examples.png" description="일반적인 이미지 인식 데이터셋 예시: CIFAR-10" class="large-image" %}

### 이미지 인식 문제의 정립: Classification, Detection, Segmentation

연구실 차원에서의 이런 올망졸망한(?) 벤치마크 데이터셋에서 출발하여, 그 후에는 1만 장 이상의 거대한 스케일의 이미지 데이터셋에 대하여 인식 성능을 겨루는 대회가 본격적으로 등장하였습니다. 초창기의 이미지 인식 대회 중 가장 대표적인 것이 *PASCAL VOC Challenge*입니다. 이 대회를 기점으로, 이미지 인식에서 다루는 문제들이 어느 정도 정형화되었다고 할 수 있습니다.

{% include image.html name="image-recognition-overview" file="classification-detection-segmentation.png" description="PASCAL VOC Challenge 문제: Classification, Detection, Segmentation" class="large-image" %}

PASCAL VOC Challenge를 기준으로 볼 때, 이미지 인식 분야에서 다루는 주요 문제를 크게 3가지로 정리할 수 있습니다. **Classification**, **Detection**, **Segmentation**이 바로 그것입니다. 지금부터 이들 각각의 문제가 구체적으로 무엇인지, 과거에는 이들 문제에 어떻게 접근했는지, 오늘날 딥러닝 기술을 적용하여 이들 문제에 어떻게 접근하고 있는지의 순으로 이야기해 보도록 하겠습니다.

## Classification

### 문제 정의

Classification 문제에서는, *주어진 이미지 안에 어느 특정한 카테고리에 해당하는 사물이 포함되어 있는지 여부를 분류*하는 것을 주요 목표로 합니다. 이 때, 분류의 대상이 되는 이미지에는 하나의 사물만이 포함되어 있거나, 또는 복수 개의 서로 다른 사물들이 포함되어 있을 수도 있습니다. 

본격적인 Classification을 수행하기 전에, 관심의 대상이 되는 전체 카테고리들을 미리 정해놓고 시작해야 합니다. 이러한 카테고리 하나하나를 **클래스(class)**라고 부릅니다. 예를 들어, PASCAL VOC Challenge에서는 총 20가지 클래스를 상정하고, 이에 대한 classification을 수행하도록 하였습니다.

{% include image.html name="image-recognition-overview" file="pascal-voc-classes.png" description="PASCAL VOC Challenge에서 다루는 20가지 클래스" class="full-image" %}

만약 모든 이미지가 반드시 하나의 사물만을 포함하도록 전제되어 있다면, 분류를 수행하는 모델을 '주어진 이미지에는 카테고리 X가 포함되어 있을 것이다'는 식의 결론을 내도록 디자인하는 것이 일반적입니다. 반면 이미지 상에 복수 개의 사물들이 포함되어 있을 수 있다면, 분류 모델을 '주어진 이미지에는 카테고리 X, Y, Z가 포함되어 있을 것이다'는 식의 결론을 내도록 디자인합니다. 보통 전자의 경우가 좀 더 쉬운 문제로 취급되며(e.g. MNIST, CIFAR-10), 후자가 좀 더 어려운 문제로 취급되는 경우가 많습니다(e.g. PASCAL VOC).

Classification 문제는, 이어질 Detection 및 Segmentation 문제를 향한 출발점이라고 할 수 있습니다. Detection 및 Segmentation 문제 해결을 위해서는 특정 클래스에 해당하는 사물이 이미지 상의 어느 곳에 위치하는지에 대한 정보를 파악해야 하는데, 이를 위해서는 우선 그러한 사물이 이미지 상에 존재하는지 여부가 반드시 먼저 파악되어야 하기 때문입니다. 이러한 경향 때문에, Classification 문제에서 우수한 성능을 발휘했던 모델을 Detection 또는 Segmentation을 위한 구조로 변형하여 사용할 경우, 그 역시 상대적으로 우수한 성능을 발휘하는 경향이 있습니다.

### 평가 척도

어떤 모델의 Classification 성능을 평가하고자 할 때, 다양한 종류의 **평가 척도(evaluation measure)** 중 하나 혹은 여러 개를 선정하여 사용할 수 있습니다. 일반적으로 가장 쉽게 떠올릴 수 있는 척도로 **정확도(accuracy)**가 있습니다. Classification 문제에서의 정확도는 일반적으로, 테스트를 위해 주어진 전체 이미지 수 대비 올바르게 분류한 이미지 수로 정의합니다. 

\begin{equation}
\text{accuracy} = \frac{\text{올바르게 분류한 이미지 수}} {\text{전체 이미지 수}}
\end{equation}

만약 하나의 이미지에 하나의 사물만이 포함되어 있다는 것이 전제되어 있다면, 위와 같이 정의된 정확도를 평가 척도로 즉각 사용하여도 크게 문제가 없습니다. 그런데 만약 하나의 이미지에 복수 개의 사물들이 포함되어 있을 수 있다면, 위의 정확도를 그대로 사용하기 곤란해지는 상황이 발생합니다. 

이러한 경우, 정확도를 각 클래스 별로 계산한 뒤 이들 전체의 대푯값(representative value)을 취하는 방식을 채택할 수 있습니다. 예를 들어, 전체 $$C$$개 클래스의 평균 정확도를 계산하고자 할 시, 아래와 같은 공식을 사용할 수 있습니다(이 때, '클래스 $$c$$ 이미지'란 클래스 $$c$$에 해당하는 사물을 포함하고 있는 이미지를 지칭합니다).

\begin{equation}
\text{평균 accuracy} = \frac{1}{C} \sum_{c=1}^{C} \frac{\text{올바르게 분류한 클래스 c 이미지 수}} {\text{전체 클래스 c 이미지 수}}
\end{equation}

그런데, 보다 현실적인 Classification 문제에서는 단순한 정확도 대신 *<a href="https://en.wikipedia.org/wiki/Precision_and_recall" target="_blank">정밀도(precision)</a>*, *<a href="https://en.wikipedia.org/wiki/Precision_and_recall" target="_blank">재현율(recall)</a>*, *<a href="https://en.wikipedia.org/wiki/F1_score" target="_blank">F1 score</a>* 등의 척도를 더 많이 사용합니다(이들에 대한 자세한 내용은, 각 단어에 연결된 링크를 참조하시길 바랍니다). 

### 대표적인 딥러닝 모델

- 과거 접근 방법론(TBD)
- 최근 접근 방법론
  - ResNeXt, DenseNet, DPN
  - TODO


## Detection

### 문제 정의

Detection 문제에서는, *주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 (만약 있다면) 어느 위치에 포함되어 있는지 '박스 형태'로 검출*하는 것을 목표로 합니다. 이는 특정 클래스의 사물이 포함되어 있는지 여부만을 분류하는 Classification 문제의 목표에서 한 발 더 나아간 것이라고 할 수 있습니다. '위치 파악'이라는 의미를 부각하기 위해, 다른 이미지 인식 대회에서는 Detection 문제를 'Image Localization' 문제라고 표현하기도 합니다. 

Detection에서 '박스 형태'로 위치를 표시한다고 하였는데, 이 때 사용하는 박스는 '네 변이 이미지 상에서 수직/수평 방향을 향한(axis-aligned)' 직사각형 모양의 박스입니다. 즉, 아래 그림과 같은 형태의 박스를 지칭하며, 이를 *바운딩 박스(bounding box)*라고 부릅니다. 

{% include image.html name="image-recognition-overview" file="pascal-voc-detection-image-example.png" description="Detection 문제에서의 바운딩 박스" class="large-image" %}

바운딩 박스를 정의하기 위해서는, 전체 이미지 상에서 박스의 좌측 상단의 좌표 $$(x_1, y_1)$$과, 우측 하단의 좌표 $$(x_2, y_2)$$를 결정해야 합니다. 이와 더불어, 제시한 바운딩 박스 안에 어떤 카테고리에 해당하는 사물이 포함되어 있을지에 대한 결론도 함께 제시해야 합니다. 즉, 제시해야 하는 결론을 종합해보면 '바운딩 박스 $$(x_1, y_1, x_2, y_2)$$ 안에는 카테고리 X가 포함되어 있을 것이다'는 식으로 표현할 수 있습니다. 

여기에서도 마찬가지로, 만약 모든 이미지가 반드시 하나의 사물만을 포함하도록 전제되어 있다면, 검출을 수행하는 모델은 위의 결론을 하나만 내도록 디자인하면 됩니다. 반면 이미지 상에 복수 개의 사물들이 포함되어 있을 수 있다면, 검출 모델을 '1번 바운딩 박스 $$(x_1^{(1)}, y_1^{(1)}, x_2^{(1)}, y_2^{(1)})$$ 안에는 카테고리 X가, 2번 바운딩 박스 $$(x_1^{(2)}, y_1^{(2)}, x_2^{(2)}, y_2^{(2)})$$ 안에는 카테고리 Y가, 3번 바운딩 박스 $$(x_1^{(3)}, y_1^{(3)}, x_2^{(3)}, y_2^{(3)})$$ 안에는 카테고리 Z가 포함되어 있을 것이다'는 식의 결론을 내도록 디자인해야 합니다. 

단순하게 생각해볼 때, Classification에 비해 바운딩 박스들과 관련된 정보를 추가로 제시해야 하고, 이들 각각에 결부된 사물의 클래스에 대한 분류를 빠짐없이 수행해야 한다는 측면에서, Detection 문제는 더 높은 난이도를 지닙니다. 

### 평가 척도

- 평가 척도: IOU 기준, 사전에 지정된 threshold를 초과하는지

*TODO: IOU와 threshold에 대한 이미지 추가*

- 과거 접근 방법론(TBD)
- 최근 접근 방법론
  - Region Proposals(e.g. R-CNN 계열)
  - YOLO, SSD


## Segmentation

- 개념적으로는, 픽셀 단위로 classification을 한 것
- Semantic Segmentation: 이미지 상의 모든 픽셀을 대상으로 분류를 수행함(이 때, 서로 다른 사물이더라도 동일한 카테고리에 해당한다면, 서로 동일한 것으로 분류함)
- Instance Segmentation: 사물 카테고리 단위가 아니라, 사물 단위로 픽셀 별 분류를 수행함
- 평가 척도: IOU
- 과거 접근 방법론(TBD)
- 최근 접근 방법론
  - Fully convolutional networks(e.g. FCN)


## 결론

- TODO

## References

- 오일석, \<컴퓨터 비전\>, 한빛아카데미, 2014
- 특수한 이미지 인식 문제 예시: 얼굴 인식(FERET database)
  - <a href="https://www.researchgate.net/profile/Amnart_Petpon/publication/221412223_Face_Recognition_with_Local_Line_Binary_Pattern/links/0fcfd508e345a96d26000000/Face-Recognition-with-Local-Line-Binary-Pattern.pdf" target="_blank">Petpon, Amnart, and Sanun Srisuk. "Face recognition with local line binary pattern." Image and Graphics, 2009. ICIG'09. Fifth International Conference on. IEEE, 2009.</a>
- 특수한 이미지 인식 문제 예시: 필적 인식(MNIST database)
  - <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">LeCun, Yann et al. "THE MNIST DATABASE" http://yann.lecun.com/exdb/mnist. Accessed 14 November 2017.</a>
- 일반적인 이미지 인식 문제 데이터셋 예시: CIFAR-10
  - <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009).</a>
- PASCAL VOC 문제: Classification, Detection, Segmentation
  - <a href="http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf" target="_blank">Everingham, Mark, et al. "The pascal visual object classes challenge: A retrospective." International journal of computer vision 111.1 (2015): 98-136.</a>
- 정밀도(precision)와 재현율(recall)
  - <a href="https://en.wikipedia.org/wiki/Precision_and_recall" target="_blank">"Precision and recall." Wikipedia contributors. "Precision and recall." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 3 Oct. 2017. Web. 24 Nov. 2017.</a>
- F1 score
  - <a href="https://en.wikipedia.org/wiki/F1_score" target="_blank">Wikipedia contributors. "F1 score." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 29 Oct. 2017. Web. 24 Nov. 2017. </a>
