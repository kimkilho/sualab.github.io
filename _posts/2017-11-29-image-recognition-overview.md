---
layout: post
title: "(가제)이미지 인식 문제의 기본 접근 방법"
date: 2017-11-05 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, machine-vision]
tags: [machine-learning, data-science, machine-vision]
comments: true
name: image-recognition-overview
---

지난 번 글까지 해서 수아랩의 핵심 기술들 중 하나인 '딥러닝'에 대해 알아보았습니다. 오늘날 딥러닝 기술이 적용되고 있는 분야는 이미지 인식, 음성 인식, 자연어 처리 등 여러 가지가 있습니다. 오늘은 이러한 적용 분야들 중, 딥러닝의 위력을 가장 드라마틱하게 보여주고 있다고 할 수 있는 '이미지 인식' 분야에서 다루는 문제들에 대하여 살펴보고자 합니다. 

## 서론

**이미지 인식(image recognition)** 문제에서는, 기계로 하여금 주어진 이미지 상에 포함되어 있는 대상이 *무엇인지*, 또한 *어느 위치에 있는지* 파악하도록 하는 것을 주된 목표로 합니다. 예를 들어, 수아랩 기술 블로그를 오랫동안 보아 오셨다면 너무나도 친숙할 만한, 아래와 같은 이미지가 주어졌다고 합시다.

{% include image.html name="image-recognition-overview" file="tree-image.png" description="인간이 받아들이는 나무 이미지" class="large-image" %}

5살 남짓의 어린 아이조차도, 위 이미지를 관찰한 순간 그 안에 '나무'라는 대상이 포함되어 있다는 것을 불과 0.1초 내로 *빠르고 정확하게* 인식할 수 있습니다. 비단 나무뿐만 아니라, 어린 아이는 그 주변에 존재하는 다양한 대상들에 대해서도 큰 무리 없이 유사한 속도와 성능(?)으로 인식할 것이라고 쉽게 예상할 수 있습니다.

그러나 오늘날 과학 기술이 꽃을 피운 21세기에 접어들었음에도 불구하고, 이렇게 어린 아이조차도 쉽게 할 수 있는 이미지 인식이, 기계에게는 여전히 매우 어려운 일로 받아들여지고 있습니다. 지난 <a href="http://research.sualab.com/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a> 글에서도 언급하였듯이, 기계는 이미지를 **픽셀(pixel)** 단위의 수치화된 형태로 받아들이며, 일반적으로 인간이 보고 이해할 수 있을만큼 큰 이미지는 매우 많은 수의 픽셀들로 구성되어 있습니다. 

{% include image.html name="image-recognition-overview" file="tree-image-pixels.svg" description="기계가 받아들이는 나무 이미지: 수많은 픽셀을 통한 표현*<br><small>(*주의: 격자 안의 하나의 정사각형의 크기는 실제 1픽셀보다는 크며, 설명을 돕기 위해 과장하였습니다.)</small>" class="large-image" %}

> 위 나무 이미지는, 실제로는 756x409(=309,204)개의 픽셀로 이루어져 있습니다.

위와 같은 이미지를 보고 '나무'라는 추상적인 개념을 뽑아내는 작업에 있어, 인간의 경우 (아직 완전히 밝혀지지 않은 모종의 매커니즘에 의해) '선택적 주의 집중(selective attention)' 및 '문맥(context)'에 기반한 '종합적 이해' 등의 과정을 거치며, 이 작업을 *직관적으로* 빠른 속도로 정확하게 수행할 수 있습니다. 반면, 기계는 '선택적 주의 집중' 능력이 없기 때문에 픽셀의 값을 빠짐없이 하나하나 다 살펴봐야 하므로 일단 이 과정에서 속도가 느려질 수밖에 없으며, 이렇게 읽어들인 픽셀로부터 어떻게 '문맥' 정보를 추출하고, 또 이들을 어떻게 '종합하고 이해'하는 것이 최적인지도 알지 못하므로 그 성능 또한 인간에 한참 뒤떨어질 수밖에 없습니다.

이러한 상황에서, 기계의 이미지 인식 속도와 성능을 인간의 수준으로 끌어올리기 위한 가장 효과적인 방법은 '인간이 이미지를 인식하는 매커니즘을 밝혀내고, 이를 기계로 하여금 모방하도록 해 보자'는 것이라고 생각할 수 있습니다. 실제로, 이는 뇌 과학(brain science) 분야에서 주로 다루어지는 연구 주제입니다. 이를 위해서는 인간의 지능을 구성하는 지식 표현, 학습, 추론, 창작 등에 해당하는 인공지능 문제들이 모두 풀려야 가능할 것으로 보이나, 아직 갈 길이 한참 먼 것이 현실입니다.

한편 이미지 인식 연구 초창기에 뇌 과학의 연구 성과를 마냥 기다릴 수만은 없었던 공학자들은, 인간의 인식 매커니즘을 그대로 모방하려는 시도 대신, 기존의 이미지 인식 문제의 범위를 좁혀서 좀 더 특수한 목적을 지니는 쉬운 형태의 문제로 치환하고 이들을 수학적 기법을 통해 해결하는 방법을 고안해 왔습니다. 예를 들어, 인간의 '선택적 주의 집중' 및 '문맥 파악' 능력에는 못 미치지만, 어떤 특수한 문제 해결에 효과적인 **요인(feature)**을 정의하여 사용하고, 이들을 '종합하고 이해'하도록 하기 위해 **러닝 모델(learning model)**과 **러닝 알고리즘(learning algorithm)**을 사용하여 이를 머신러닝 차원으로 해결하고자 하였습니다.

{% include image.html name="image-recognition-overview" file="face-recognition-examples.png" description="특수한 이미지 인식 문제 예시: 얼굴 인식" class="large-image" %}


- PASCAL VOC, ImageNet ILSVRC 등의 대회 종목 및 규정을 기준으로 함
- 과거의 전통적인 머신러닝 기반 접근 방법론, 최근 딥러닝 기반 접근 방법론을 모두 소개

## Classification

- 하나의 이미지에는 하나의 사물이 포함되어 있다고 전제함
- 이미지 안에 포함되어 있는 사물이 전체 N개의 카테고리들 중 어떤 것어 해당하는지 분류하는 문제
- 이미지 인식 문제의 가장 기본이 되며, Segmentation, Localization/Detection 등의 문제를 위한 출발점
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - **TODO**

## Segmentation

- 개념적으로는, 픽셀 단위로 classification을 한 것
- Semantic Segmentation: 이미지 상의 모든 픽셀을 대상으로 분류를 수행함(이 때, 서로 다른 사물이더라도 동일한 카테고리에 해당한다면, 서로 동일한 것으로 분류함)
- Instance Segmentation: 사물 카테고리 단위가 아니라, 사물 단위로 픽셀 별 분류를 수행함
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - Fully convolutional networks(e.g. FCN)

## Localization/Detection

- Localization: Classification + 대략적 위치 파악
- Object Detection: 복수 개의 사물에 대한 Localization
- CNN with multitask loss: classification + bounding box regression
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - Region Proposals(e.g. R-CNN 계열)
  - YOLO, SSD

## 결론

- TODO

## References

- 오일석, \<컴퓨터 비전\>, 한빛아카데미, 2014
- 이미지넷 ILSVRC 측에서 발표한 논문
  - <a href="https://arxiv.org/pdf/1409.0575.pdf" target="_blank">Russakovsky, Olga, et al. "Imagenet large scale visual recognition challenge." International Journal of Computer Vision 115.3 (2015): 211-252.</a>
- 특수한 이미지 인식 문제 예시: 얼굴 인식
  - <a href="https://www.researchgate.net/profile/Amnart_Petpon/publication/221412223_Face_Recognition_with_Local_Line_Binary_Pattern/links/0fcfd508e345a96d26000000/Face-Recognition-with-Local-Line-Binary-Pattern.pdf" target="_blank">Petpon, Amnart, and Sanun Srisuk. "Face recognition with local line binary pattern." Image and Graphics, 2009. ICIG'09. Fifth International Conference on. IEEE, 2009.</a>
