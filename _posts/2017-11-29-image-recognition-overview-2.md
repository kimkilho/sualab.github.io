---
layout: post
title: "이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로 (2)"
date: 2017-11-29 10:00:00 +0900
author: kilho_kim
categories: [computer-vision]
tags: [pascal voc, classification, detection, segmentation]
comments: true
name: image-recognition-overview-2
---

[(이전 포스팅 보기)]({{ page.url }}/../image-recognition-overview-1.html)

## Detection

### 문제 정의

Detection 문제에서는, *주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 (만약 있다면) 어느 위치에 포함되어 있는지 '박스 형태'로 검출하는 모델을 만드는 것*을 목표로 합니다. 이는 특정 클래스의 사물이 포함되어 있는지 여부만을 분류하는 Classification 문제의 목표에서 한 발 더 나아간 것이라고 할 수 있습니다. '위치 파악'이라는 의미를 부각하기 위해, 다른 이미지 인식 대회에서는 Detection 문제를 'Image Localization' 문제라고 표현하기도 합니다. 

Detection에서 '박스 형태'로 위치를 표시한다고 하였는데, 이 때 사용하는 박스는 네 변이 이미지 상에서 수직/수평 방향을 향한(axis-aligned) 직사각형 모양의 박스입니다. 이는 아래 그림과 같은 형태의 박스를 지칭하며, 이를 **바운딩 박스(bounding box)**라고 부릅니다. 

{% include image.html name=page.name file="pascal-voc-detection-bbox-example.svg" description="Detection 문제에서의 바운딩 박스 예시<br><small>(예시 이미지: VOC2009 데이터셋 - 2009_002093.jpg)</small>" class="medium-image" %}

바운딩 박스를 정의하기 위해서는, 전체 이미지 상에서 박스의 좌측 상단의 좌표 $$(x_1, y_1)$$과, 우측 하단의 좌표 $$(x_2, y_2)$$를 결정해야 합니다. 이와 더불어, 제시한 바운딩 박스 안에 포함된 사물에 대한 각 클래스 별 신뢰도 점수도 함께 제시해야 합니다. 즉, '바운딩 박스 $$(x_1, y_1, x_2, y_2)$$ 안에 클래스 X의 사물이 존재할 가능성이 $$s_X$$이다'는 식의 결과물을 제출해야 합니다.

여기에서도 마찬가지로, 만약 '*단일 사물 검출*' 문제를 전제한다면, 수행하는 모델은 바운딩 박스 하나에 대한 결과물만을 내도록 디자인하면 됩니다. 반면 '*복수 사물 검출*' 문제를 전제한다면, 검출 모델로 하여금 '1번 바운딩 박스 $$(x_1^{(1)}, y_1^{(1)}, x_2^{(1)}, y_2^{(1)})$$ 안에는 {클래스 X: $$s_X^{(1)}$$, 클래스 Y: $$s_Y^{(1)}$$, ...}, 2번 바운딩 박스 $$(x_1^{(2)}, y_1^{(2)}, x_2^{(2)}, y_2^{(2)})$$ 안에는 {클래스 X: $$s_X^{(1)}$$, 클래스 Y: $$s_Y^{(1)}$$, ...}, 3번 바운딩 박스 $$(x_1^{(3)}, y_1^{(3)}, x_2^{(3)}, y_2^{(3)})$$ 안에는 {클래스 X: $$s_X^{(1)}$$, 클래스 Y: $$s_Y^{(1)}$$, ...}'의 형태로, 복수 개의 바운딩 박스에 대한 결과물을 제출하도록 디자인해야 합니다. 

{% include image.html name=page.name file="detection-model.svg" description="Detection 문제(복수 사물 검출 문제)<br><small>(예시 이미지: VOC2009 데이터셋 - 2009_002807.jpg)</small>" class="full-image" %}

단순하게 생각해볼 때, Classification에 비해 바운딩 박스들과 관련된 정보를 추가로 제시해야 하고, 이들 각각에 결부된 사물의 클래스에 대한 분류를 빠짐없이 수행해야 한다는 측면에서, Detection 문제는 더 높은 난이도를 지닙니다. 

### 평가 척도

#### IOU(intersection over union)

Detection 문제의 경우, 사물의 클래스 및 위치에 대한 예측 결과를 동시에 평가해야 하기 때문에, 사물의 실제 위치를 나타내는 '*실제(ground truth; 이하 GT)*' 바운딩 박스 정보가 이미지 레이블 상에 포함되어 있습니다. 검출 모델의 경우 복수 개의 예측 바운딩 박스를 제출할 수 있기 때문에, 이들 중 어떤 것을 GT 바운딩 박스와 매칭시킬지에 대한 규정이 마련되어 있습니다.

이를 위해, 각 예측 바운딩 박스 $$B_p$$와 GT 바운딩 박스 $$B_{gt}$$에 대하여, 아래와 같이 정의되는 **IOU(intersection over union)**를 사용하여 $$B_p$$와 $$B_{gt}$$가 서로 얼마나 '겹쳐지는지'를 평가합니다.

\begin{equation}
B_p\text{와 } B_{gt}\text{ 의 IOU} = \frac{B_p \cap B_{gt} \text{ 영역 넓이}} {B_p \cup B_{gt} \text{ 영역 넓이}}
\end{equation}

{% include image.html name=page.name file="bbox-overlap.svg" description="$$B_p$$와 $$B_{gt}$$ 간의 IOU 계산" class="full-image" %}

PASCAL VOC Challenge에서는, 예측 바운딩 박스와 GT 바운딩 박스 간의 IOU에 대한 문턱값을 $$0.5$$로 정해 놓고 있습니다. 즉, *예측 바운딩 박스와 GT 바운딩 박스 간에 겹친 영역의 비율이 50%를 넘겼을 때만, 두 바운딩 박스를 매칭한 뒤, 해당 예측 바운딩 박스의 신뢰도 점수를 평가*하는 방식을 채택합니다.

이 때 주의할 점은, 하나의 GT 바운딩 박스에 대하여 여러 개의 예측 바운딩 박스가 모두 IOU를 50% 넘겨 매칭된 경우, 이들 모두 결과적으로는 매칭에 실패한 것으로 간주되어 채점 대상에서 누락된다는 점입니다. 예측 바운딩 박스와 GT 바운딩 박스 간의 매칭 성사 및 실패 사례를 아래 그림에서 나타내었습니다.

{% include image.html name=page.name file="bbox-overlap-examples.svg" description="$$B_p$$와 $$B_{gt}$$ 간의 매칭 사례<br><small>(예시 이미지: VOC2010 데이터셋 - 2010_000413.jpg)</small>" class="large-image" %}

> 이러한 규정 때문에, '일단 마구 질러보고, 하나만 얻어 걸려라'는 식의 전략은 지양하는 것이 좋겠습니다.

#### 정밀도와 재현율

일단 위와 같이 하나의 GT 바운딩 박스 당 하나의 예측 바운딩 박스가 매칭된 이후에는, 해당 예측 바운딩 박스에 결부된 신뢰도 점수를 기반으로 정밀도 또는 재현율을 계산합니다. 이 과정은 Classification 문제에서와 거의 동일한 방법으로 진행되므로, 자세한 설명은 생략하도록 하겠습니다.

### 의의

Detection 문제는 근본적으로 Classification 문제보다 어려운 문제입니다. 실제로 오늘날 Classification 문제를 위해 개발된 분류 모델들의 성능 발전 수준에 비해, Detection 문제를 위해 개발된 검출 모델들의 성능 발전 수준이 상대적으로 뒤처져 있는 것이 사실입니다. 

그럼에도 불구하고, Detection 문제는 사물의 위치에 대한 예측 정보까지 추가로 제공해 준다는 측면에서, 다양한 산업 현장에 적용될 수 있는 잠재적인 가치를 지니고 있습니다. 실제 산업에의 적용을 위해, Detection 문제를 연구하는 사람들은 검출 결과의 성능 자체를 향상시키기 위한 노력과 더불어, 검출 모델의 이미지 1장 당 처리 소요 시간을 낮추기 위한 노력도 병행하고 있습니다.


## Segmentation

### 문제 정의

Segmentation 문제에서는, *주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 (만약 있다면) 어느 위치에 포함되어 있는지 '픽셀 단위로' 분할하는 모델을 만드는 것*을 목표로 합니다. 이는 사물의 위치를 바운딩 박스로 표시하는 Detection 문제보다 더 자세하게 위치를 표시해야 하기 때문에, Detection 문제보다 더 어려운 문제에 해당합니다.

{% include image.html name=page.name file="segmentation-model.svg" description="Segmentation 문제<br><small>(예시 이미지: VOC2007 데이터셋 - 2007_001423.jpg)</small>" class="full-image" %}

Segmentation 문제는, 그 정의 상 본질적으로 *'픽셀'들을 대상으로 한 Classification 문제*라고 해도 크게 무리가 없습니다. 주어진 이미지 내 각 위치 상의 픽셀들을 하나씩 조사하면서, 현재 조사 대상인 픽셀이 어느 특정한 클래스에 해당하는 사물의 일부인 경우, 해당 픽셀의 위치에 그 클래스를 나타내는 '값'을 표기하는 방식으로 예측 결과물을 생성합니다. 만약 조사 대상 픽셀이 어느 클래스에도 해당하지 않는 경우, 이를 '*배경(background)*' 클래스로 규정하여 예측 결과물의 해당 위치에 $$0$$을 표기합니다. 이렇게 생성된 결과물을 **마스크(mask)**라고도 부릅니다.

가령 'person' 클래스를 나타내는 값이 $$1$$이라고 했다면, 아래 그림과 같이 사람이 포함된 이미지를 받았을 때 실제 사람이 위치하는 'person' 영역에만 $$1$$을, 그렇지 않은 'background' 영역에는 $$0$$을 기재할 수 있습니다. 이런 식으로 하여, 'dog', 'pottedplant', 'motorbike' 등의 다른 클래스에 대해서도, $$2$$, $$3$$, $$4$$ 등 해당 클래스들을 나타내는 값을 표기하는 방식으로 예측 마스크를 생성할 수 있습니다.

{% include image.html name=page.name file="segmentation-result-to-values.svg" description="Segmentation 분할 결과 = 픽셀 단위 분류 결과<br><small>(좌측 이미지는 PASCAL VOC Challenge의 마스킹 규칙에 맞게 빨간색으로 채색하였으며,<br>우측 결과물과 실제로 완벽하게 일치하지는 않습니다.)</small>" class="large-image" %}

(하지만 엄밀하게 말하자면, Classification 문제에서는 각 이미지에 대한 신뢰도 점수를 제출하도록 요구한다면, Segmentation 문제에서는 각 픽셀이 어떤 클래스에 해당하는지를 나타내는 값을 곧바로 제출하도록 한다는 점에서 차이가 있다고 할 수 있겠습니다.)

#### 세부 문제 구분

Segmentation 문제는 **Semantic Segmentation**과 **Instance Segmentation**의 두 가지 세부 문제로 구분할 수 있습니다. Semantic Segmentation은 분할의 기본 단위를 클래스로 하여, 동일한 클래스에 해당하는 사물을 예측 마스크 상에 동일한 색상으로 표시합니다. 반면 Instance Segmentation은 분할의 기본 단위를 사물로 하여, 동일한 클래스에 해당하더라도 서로 다른 사물에 해당하면 이들을 예측 마스크 상에 다른 색상으로 표시합니다. 아래 그림에서 Semantic Segmentation과 Instance Segmentation 간의 차이를 극명하게 확인할 수 있습니다.

{% include image.html name=page.name file="segmentation-types.svg" description="Segmentation의 종류: Semantic Segmentation, Instance Segmentation<br><small>(예시 이미지: VOC2007 데이터셋 - 2007_000129.jpg)</small>" class="large-image" %}

### 평가 척도

#### IOU(intersection over union)

Segmentation 문제도 Detection 문제의 경우와 유사하게, 사물의 실제 위치를 나타내는 '*실제(GT)*' 마스크가 이미지 레이블 상에 포함되어 있습니다. 예측 마스크의 특정 클래스를 나타내는 영역 $$A_p$$와 GT 마스크의 해당 클래스 영역 $$A_{gt}$$에 대하여, 아래의 **IOU(intersection over union)**를 사용하여 $$A_p$$와 $$A_{gt}$$가 서로 얼마나 '겹쳐지는지'를 평가합니다.

\begin{equation}
A_p\text{와 } A_{gt}\text{ 의 IOU} = \frac{A_p \cap A_{gt} \text{ 영역 넓이}} {A_p \cup A_{gt} \text{ 영역 넓이}}
\end{equation}

{% include image.html name=page.name file="segmentation-iou.svg" description="$$A_p$$와 $$A_{gt}$$ 간의 IOU 계산<br><small>(마스크 상의 흰 색으로 표시된 픽셀들의 경우, IOU 계산 시 고려 대상에서 제외됨,<br>예시 이미지: VOC2007 데이터셋 - 2007_001458.jpg)</small>" class="full-image" %}

PASCAL VOC Challenge에서는, 각 클래스 별로 위와 같이 계산된 IOU 자체를 최종 평가 척도로 사용합니다. 이 때 주의할 점은, 기존 GT 마스크 상에서 ('background' 클래스를 제외한) 특정 클래스를 나타내는 영역 $$A_{gt}$$의 가장자리에는 반드시 *폭 5px의 '흰색' 경계선*이 표시되어 있다는 것입니다. 이렇게 GT 마스크 상에서 흰색으로 표시된 픽셀의 경우, *IOU 계산 시 고려 대상에서 완전히 배제*됩니다. 보통 이미지 상의 사물을 둘러싼 '정확한' 경계를 결정하는 과정에서는 사람들 사이에서도 의견이 분분할 수밖에 없는데, 이러한 애매함을 해결하고자 도입한 규정이라고 보면 됩니다.

> 실제로 이미지 인식 문제의 '정답' 마스크 혹은 레이블을 만드는 과정에서, 이런 '애매성(ambiguity)'이 문제가 되는 경우가 굉장히 많습니다. 이미지 상에서 애매성의 원인이 될 만한 부분들에 대한 일관된 처리 규정을 미리 정해 놓아야, 모델 입장에서 억울할 만한(?) 성능 저하를 방지할 수 있습니다.

### 의의

Segmentation 문제는 모든 픽셀에 대하여 클래스 분류를 수행해야 한다는 점에서, 성능 발전 수준도 다른 문제에 비해 상대적으로 낮은 편이며, 분류 모델의 이미지 1장 당 처리 소요 시간도 매우 긴 편입니다. 하지만, 사물의 위치에 대한 정교한 인식 결과물을 얻는 것이 가장 중요한 문제 상황에서는, 이를 Segmentation 문제로 가정하고 이를 해결하기 위한 분할 모델을 개발하는 것이 최선의 방법이라고 할 수 있겠습니다. 


## 결론

이미지 인식 문제에서는, 기계로 하여금 주어진 이미지 상에 포함되어 있는 대상이 무엇인지, 또한 어느 위치에 있는지 등을 파악하도록 하는 것을 주된 목표로 합니다. 그러나 인간이 빠르고 정확하게 할 수 있는 이미지 인식은, 기계에게는 매우 어려운 일로 받아들여져 왔습니다.  초창기에는 공학자들을 중심으로, 얼굴 인식, 필적 인식 등 특수한 목적을 지니는 문제에 대하여 머신러닝 방법론에 기반한 제한적인 이미지 인식을 시도해 왔습니다. 근래에 접어들면서 PASCAL VOC Challenge 등 거대한 스케일의 이미지 인식 대회가 등장하기 시작하였으며, 이 때부터 이미지 인식 문제가 상당 부분 정형화되면서 이들에 대한 연구가 활발하게 진행되어 왔습니다. 

Classification 문제에서는, 주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 포함되어 있는지 여부를 분류하는 모델을 만드는 것을 주요 목표로 합니다. 분류 모델은 주어진 이미지에 대하여 신뢰도 점수 결과물을 제출하며, 이에 기반하여 단일 사물 분류 문제에서는 정확도를, 복수 사물 분류 문제에서는 정밀도 및 재현율을 주요한 평가 척도로 사용합니다. 정밀도 및 재현율에 대한 문턱값을 미리 정하여 최종 성능을 산출하는 대신, PASCAL VOC Challenge에서는 평균 정밀도를 채택하여 모든 가능한 재현율 값에 대한 평균적인 정밀도를 계산합니다.

Detection 문제에서는, 주어진 이미지 안에 어느 특정한 클래스에 해당하는 사물이 어느 위치에 포함되어 있는지 바운딩 박스 형태로 검출하는 모델을 만드는 것을 목표로 하며, Segmentation 문제에서는 이를 픽셀 단위로 분할하는 모델을 만드는 것을 목표로 합니다. Detection 문제에서는 검출 모델의 성능 측정을 위해, 모델이 예측한 바운딩 박스와 GT 바운딩 박스를 매칭하는 과정을 선행하는데, 이 때 IOU를 사용하여 둘 간에 얼마나 겹쳐지는지를 평가합니다. Segmentation 문제에서의 분할 모델은 클래스를 나타내는 값들로 구성되는 마스크를 제출하도록 디자인되며, IOU를 사용하여 예측 성능에 대한 측정이 이루어집니다.

지금까지 딥러닝 기술의 주요 적용 분야인 이미지 인식 문제의 개요를 살펴보았으므로, 이제 여러분들은 이미지 인식 문제에 적용되는 주요 딥러닝 모델에 대해 탐구할 준비가 되셨을 것으로 생각합니다. \*바로 다음 글에서부터, Classification, Detection, Segmentation 각각의 문제에 적용된 바 있는 대표적인 딥러닝 모델에 대해, 순서대로 상세하게 알아보도록 하겠습니다.


## References

- 오일석, \<컴퓨터 비전\>, 한빛아카데미, 2014
- 특수한 이미지 인식 문제 예시: 얼굴 인식(FERET database)
  - <a href="https://www.researchgate.net/profile/Amnart_Petpon/publication/221412223_Face_Recognition_with_Local_Line_Binary_Pattern/links/0fcfd508e345a96d26000000/Face-Recognition-with-Local-Line-Binary-Pattern.pdf" target="_blank">Petpon, Amnart, and Sanun Srisuk. "Face recognition with local line binary pattern." Image and Graphics, 2009. ICIG'09. Fifth International Conference on. IEEE, 2009.</a>
- 특수한 이미지 인식 문제 예시: 필적 인식(MNIST database)
  - <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">LeCun, Yann et al. "THE MNIST DATABASE" http://yann.lecun.com/exdb/mnist. Accessed 14 November 2017.</a>
- 일반적인 이미지 인식 문제 데이터셋 예시: CIFAR-10
  - <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009).</a>
- PASCAL VOC Challenge 문제: Classification, Detection, Segmentation
  - <a href="https://pdfs.semanticscholar.org/0ee1/916a0cb2dc7d3add086b5f1092c3d4beb38a.pdf" target="_blank">Everingham, Mark, et al. "The pascal visual object classes (voc) challenge." International journal of computer vision 88.2 (2010): 303-338.</a>
  - <a href="http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf" target="_blank">Everingham, Mark, et al. "The pascal visual object classes challenge: A retrospective." International journal of computer vision 111.1 (2015): 98-136.</a>
- 분류 모델 예시 그림: LeNet
  - <a href="http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf" target="_blank">LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.</a>
