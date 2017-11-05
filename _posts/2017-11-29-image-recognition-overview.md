---
layout: post
title: "이미지 인식 문제의 기본 접근 방법"
date: 2017-11-05 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, machine-vision]
tags: [machine-learning, data-science, machine-vision]
comments: true
name: image-recognition-overview
---

*Pre-서론*

## 서론

- 사람의 이미지 인식 vs 기계의 이미지 인식
  - 이미지 데이터의 기본 구성(저번 포스팅 내용 복기)
- PASCAL VOC, ImageNet ILSVRC 등의 대회 종목 및 규정을 기준으로 함
- 과거의 전통적인 머신러닝 기반 접근 방법론, 최근 딥러닝 기반 접근 방법론을 모두 소개

## Classification

- 하나의 이미지에는 하나의 사물이 포함되어 있다고 전제함
- 이미지 안에 포함되어 있는 사물이 전체 N개의 카테고리들 중 어떤 것어 해당하는지 분류하는 문제
- 이미지 인식 문제의 가장 기본이 되며, Localization/Detection, Segmentation 등의 문제를 위한 출발점
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - **TODO**

## Localization/Detection

- Localization: Classification + 대략적 위치 파악
- Object Detection: 복수 개의 사물에 대한 Localization
- CNN with multitask loss: classification + bounding box regression
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - Region Proposals(e.g. R-CNN 계열)
  - YOLO, SSD

## Segmentation

- 개념적으로는, 픽셀 단위로 classification을 한 것
- Semantic Segmentation: 이미지 상의 모든 픽셀을 대상으로 분류를 수행함(이 때, 서로 다른 사물이더라도 동일한 카테고리에 해당한다면, 서로 동일한 것으로 분류함)
- Instance Segmentation: 사물 카테고리 단위가 아니라, 사물 단위로 픽셀 별 분류를 수행함
- 과거 접근 방법론
  - **TODO**
- 최근 접근 방법론
  - Fully convolutional networks(e.g. FCN)

## 결론

## References

- <a href="https://work.caltech.edu/telecourse.html" target="_blank">Abu-Mostafa, Magdon-Ismail, Lin, \<Learning from Data\></a>
