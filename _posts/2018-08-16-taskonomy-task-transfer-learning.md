---
layout: post
title: "Taskonomy: Disentangling Task Transfer Learning 리뷰"
date: 2018-08-16 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, computer-vision]
tags: [taskonomy, visual tasks, transfer learning]
comments: true
name: taskonomy-task-transfer-learning
---

오늘날 딥러닝 분야에서 주요하게 연구되는 주제들 중 하나로 **transfer learning(전이 학습)**이 있습니다. 이는 어느 특정한 task(이하 source task; e.g. classification, detection, segmentation 등)에 대하여 학습된 딥러닝 모델을, 다른 task(이하 target task)로 '전이'하여 해당 모델을 사후적으로 학습하는 개념을 포괄합니다. 보통 특정한 task를 학습할 시 transfer learning 방법을 적용할 경우, 해당 task를 학습하기 위해 단순히 랜덤하게 초기화된 딥러닝 모델을 사용하는 것보다, 더 적은 양의 학습 데이터를 사용하면서 더 우수한 성능을 발휘하는 것으로 익히 알려져 왔습니다. 특히 딥러닝 기술이 적용되는 산업 현장에서는 학습 데이터를 구축하는 것 자체가 '비용'과 직결되기 때문에, 이 transfer learning의 연구 결과에 대해 지대한 관심을 가질 수밖에 없습니다.

지금까지 보고되어 온 transfer learning 관련 연구는, 대부분 source task와 target task를 하나씩 정해놓고 둘 간의 transferability(전이성)만을 단편적으로 판단한 경우에 해당합니다. 이러한 연구 결과들을 보다 보면, 상상력이 뛰어나신 분이라면, 자연스럽게 '둘 이상의 task들에 대한 transferability를 한 번에 펼쳐놓고 볼 수는 없을까?' 하는 생각이 들 것 같습니다. 예를 들어 어떤 데이터셋에 대하여 classification 및 detection 용 레이블을 생성하여 각각 별도의 딥러닝 모델을 학습해 놓은 상태에서, 이들을 어떻게 '동시에 적절하게' 사용하여 또 다른 segmentation 모델을 효과적이고 효율적으로 학습할 수 없을지 등을 산업 현장에서 충분히 고민할 법 할 것 같습니다.

이러한 상상에서 출발하여, 여러 개의 visual task들에 대한 딥러닝 모델의 transferability를 한 단계 추상화된 레벨에서 관계도의 형태로 나타내고, 이에 기반하여 어느 새로운 task에 대한 딥러닝 모델의 성능을 극대화하고자 할 시 어떠한 source task들을 어떻게 조합하면 될지 등을 연구한 논문이 있어 본 글에서 소개해 드리고자 합니다. Amir R. Zamir et al.의 *'Taskonomy: Disentangling Task Transfer Learning(이하 Taskonomy)*'이라는 제목의 논문으로, 올해 개최된 컴퓨터 비전 분야의 최상위 컨퍼런스인 CVPR 2018에서 Best Paper Award를 수상한 바 있습니다.

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - Image recognition(이미지 인식) 분야에서 다뤄지는 주요 문제들에 대한 기본적인 내용들을 이해하고 계신 분들
  - Deep Learning(딥러닝)의 기초적인 내용들을 이해하고 계신 분들
  - Transfer Learning에 대한 기초적인 내용들을 이해하고 계신 분들
  - 본 글에서는 일반적인 딥러닝 분야의 최신 논문들과는 다르게, 약간의 딥러닝 외적인 수학적 방법에 대한 지식을 필요로 하는 부분이 포함되어 있습니다. 본 글에서는 이러한 부분들을 최선을 다 하여(?) 설명하고자 하였으나, 설명의 편의를 위해 지나치게 디테일한 부분에 대한 설명은 생략한 경우도 있습니다. Taskonomy 논문에 대한 좀 더 자세한 내용을 확인하고 싶으시다면, <a href="https://arxiv.org/abs/1804.08328" target="_blank">이 arXiv 링크</a>를 참조해 주시길 바랍니다.
  - Taskonomy 논문의 저자들이 워낙 잘 구현해 놓은 결과물이 존재하여, 본 글에서는 Taskonomy의 구현체를 추가로 구현하지는 않았고, 리뷰 자체에만 집중하였습니다. 저자들의 구현체는 웹페이지 형태로 소개되고 있으며, <a href="http://taskonomy.stanford.edu/" target="_blank">여기</a>를 통해 확인할 수 있습니다.


## 1. Introduction

오늘날 이미지 인식 분야에서 다뤄지는 task들은 classification, depth estimation(깊이 추정), edge detection(경계선 검출), pose estimation(포즈 추정) 등 다양하게 나와 있습니다. 이러한 task를 각각 별도의 딥러닝 모델이 학습하도록 하다 보면, '서로 간에 필요로 하는 정보가 서로 유사하여 공유될 여지가 있는 task들도 있을 것 같은데, 그저 이렇게 각각 별도로 학습하는 것은 학습 데이터 및 레이블링 비용의 낭비 아닐까?'하는 의문이 들 수 있습니다. 

{% include image.html name=page.name file="task-structure-example.png" description="Taskonomy 방법을 통해 찾아낸 task structure 예시" class="large-image" %}

본 논문의 연구는 바로 이런 문제 의식에서 출발합니다. 복수 개의 서로 다른 task들 간에 잠재적으로 존재하는 이러한 관계들을 graph 형태로 구조화하여 표현하고, 이를 사용하여 어느 새로운 task에 대한 딥러닝 모델의 학습을 보다 효과적이고(성능 향상) 효율적으로(레이블링된 데이터의 양 감소) 할 수 있는 **Taskonomy** 방법을 제안하였습니다. 이 때, 학습 모델은 deep neural networks(심층 신경망)으로, 데이터셋의 종류는 이미지 데이터셋으로 한정하였습니다.

Source task에 대하여 학습한 neural network 모델의 feature representation을, target task에 그대로 적용하여 사후적으로 학습했을 때, target task를 단독으로 학습했을 경우 대비 성능 향상 수준을, 두 task들 간의 'transferability' 척도로 정의하였습니다. 이렇게 조사된 모든 task 쌍에 대한 pairwise transferability를 affinity matrix(유사도 행렬)로 표현한 후, 이로부터 일종의 Boolean Integer Programming(이진 정수 계획법) 문제를 상정하여 특정한 target task에 대한 최적의 transfer policy를 찾아내는 방식을 채택했습니다. 모든 과정은 fully computational하게 진행되며, 각 task들에 대한 prior knowledge(사전 지식)가 전혀 개입하지 않도록 구성하였습니다. 이는 모든 문제 상황에 적용 가능하도록 하는 일반성을 확보하기 위함으로 보입니다.


## 2. Related Work

Taskonomy 방법과 관련된 매우 다양한 관련 연구 주제들이 존재하며, 이들 모두 오늘날 딥러닝 연구에서 중요하게 다뤄지는 주제들이라, 각각 한 번씩 짚어볼만 합니다. 

(1) **Self-Supervised Learning**은, 레이블링 비용이 상대적으로 낮은 task로부터 학습한 정보를 활용하여, 그와 내재적으로 관련되어 있으면서 레이블링 비용이 더 높은 task에 대한 학습을 시도하는 방법들을 통칭합니다. 이는 source task를 사전에 사람이 수동으로 지정해줘야 한다는 측면에서, Taskonomy 방법과 차이가 있습니다. 

(2) **Unsupervised Learning**은, 각 task들에 대한 명시적인 레이블이 주어지지 않은 상황에서 데이터셋 자체에 공통적으로 내재되어 있는 속성을 표현하는 feature representation을 찾아내는 것과 관련되어 있습니다. Taskonomy 방법의 경우 각 task 별 레이블을 명시적으로 필요로 하며, task 상호 간의 transferability 극대화를 위한 feature representation 활용에 초점을 맞추고 있다는 측면에서 차이가 있습니다.

(3) **Meta-Learning**은, 러닝 모델 학습을 meta-level(상위 레벨)에서의 좀 더 '추상화된' 관점으로 조명하고, '학습 데이터셋의 종류를 막론하고, 모델을 좀 더 효과적으로 학습하기 위한 일반적인 방법'을 찾아내는 데 초점이 맞춰져 있습니다. 복수 개의 task들 간의 transferability를 좀 더 meta-level에서 조망하면서 이들의 structure를 찾기 위한 일반적인 방법을 제안한다는 점에서, Taskonomy 방법과 일종의 공통점이 있다고 할 수 있습니다.

(4) **Multi-Task Learning**은, 입력 데이터가 하나로 정해져 있을 때 이를 기반으로 여러 task들에 대한 예측 결과들을 동시에 출력할 수 있도록 하는 방법을 연구하는 주제입니다. 대상이 되는 task들을 동시에 커버할 수 있는 feature representation을 찾는다는 측면에서 Taskonomy 방법과 공통점이 일부 존재하나, Taskonomy 방법은 두 task들 간의 관계를 명시적으로 모델링한다는 측면에서 차이가 있습니다.

(5) **Domain Adaptation**은 transfer learning의 하나의 특수한 형태로, task는 동일하나 입력 데이터의 domain(도메인; 속성)이 크게 달라지는 경우(source domain -> target domain) 최적의 transfer policy를 찾기 위한 연구 주제입니다. Taskonomy의 경우 domain이 아닌 task가 달라지는 경우를 가정하기 때문에, 이와는 차이가 있습니다.

(6) **Learning Theoretic** 방법들은 위의 주제들과 조금씩 겹치는 부분들이 존재하며, '모델의 generalization(일반화) 성능을 담보하기 위한' 방법들에 해당합니다. 단 기존에 나와 있던 다양한 Learning Theoretic 방법들의 경우 대부분 intractable한 계산들을 포함하거나, 이러한 계산들을 배제하기 위해 모델 또는 task에 많은 제한을 둔 바 있습니다. Taskonomy 방법의 아이디어는 Learning Theoretic 방법들로부터 일부 영감을 얻었다고 할 수 있으나, 엄밀한 이론적 증명을 피하면서 좀 더 실용적인 접근을 시도한 것이라고 할 수 있습니다. 


## 3. Method

Taskonomy(task taxonomy)를 보다 엄밀하게 정의하면, '어느 task dictionary에 대하여 각 task들 간의 transferability를 담고 있는, 계산적으로 도출 가능한 directed <a href="https://en.wikipedia.org/wiki/Hypergraph" target="_blank">hypergraph</a>'라고 하고 있습니다. Taskonomy에서는 하나의 target task의 성능 극대화를 위해, 단일 source task가 아닌 여러 개의 source task들을 동시에 활용할 수 있다고 가정하기 때문에, 이를 반영하고자 이로 인해 일반적인 graph보다 좀 더 일반화된 hypergraph(하나의 edge가 복수 개의 node들을 연결할 수 있는 graph)로 정의하였다고 할 수 있습니다. 

Taskonomy 방법에 대한 본격적인 설명에 앞서, 본문에서 사용하는 주요한 notation들을 아래와 같이 일괄 정리하였습니다.

- $$\mathcal{T} = \{t_1, ..., t_n\}$$ : Target task set; Taskonomy 적용 대상 target task들의 모음
  - $$t_j$$ : Taskonomy 상에서의 $$j$$번째 target task
- $$\mathcal{S} = \{s_1, ..., s_n\}$$ : Source task set; Target task들에 대하여 활용 가능한 source task들의 모음
  - $$s_j$$ : Taskonomy 상에서의 $$i$$번째 source task
- $$k$$ : Transfer order; 어느 하나의 target task에 대하여 활용 가능한 source task의 갯수
- $$\gamma$$ : Supervision budget; Transfer에 앞서 미리 학습해 놓을 수 있는 source task들의 총 갯수(~레이블링 비용)
- $$f_t(I)$$ : Target task $$t$$의 이미지-레이블 간의 미지의 true function, 모델 학습을 통해 추정하고자 하는 대상

{% include image.html name=page.name file="taskonomy-method-overview.png" description="Taskonomy 방법 overview: Transferability 모델링 및 taxonomy 생성 과정" class="full-image" %}

Taskonomy 방법은 총 4개의 단계를 거칩니다. 1단계에서는 $$\mathcal{S}$$ 내의 각 task에 대해 특화된 모델인 task-specific network를 각각 독립적으로 학습합니다. 2단계에서는 지정된 transfer order $$k$$ 하에서, 서로 간의 조합 연산을 통해 만들어지는 source task(s) -> target task 의 각 조합 별 transferability가 수치화된 형태로 계산됩니다. 3단계에서는 앞서 계산된 transferability에 대한 normalization(정규화)를 통해 affinity matrix를 얻으며, 이를 기반으로 마지막 4단계에서는 특정한 target task에 대하여 최적의 성능을 발휘하는 transfer policy를 탐색합니다.

실험 수행 시 task dictionary 상에 명시한 task들은 총 26가지이며, 이는 computer vision 분야에서 일반적으로 다루는 문제들을 담고 있습니다. 각 task의 해결 난이도는 서로 차이가 있으며, 이에 따라 각 task 해결을 위해 사용되는 최적의 방법에 있어서도 약간의 차이가 존재합니다. 

(엄밀히 말하자면, 실험에서 선정된 26가지 task들은 모종의 task space로부터 샘플링을 통해 얻은 것들이기 때문에, 그 속성을 규명하고자 하는 task space를 완벽하게 대표한다고 보기 어려울 수도 있습니다. 이 부분을 보완하고자, 후술될 실험 결과 분석에서 기존 task dictionary에 포함되어 있지 않았던 새로운 task에 대한 일반화 성능을 검증하는 내용이 나옵니다.)

실험에 사용한 데이터셋은 저자들이 직접 제작한 것으로, 총 600가지 건물의 실내 장면이 포함되어 있는 총 400만 장의 이미지를 마련하였으며, 여기에 26가지 task에 대한 레이블링을 모두 수행하였습니다. 모든 레이블링을 사람이 진행한 것은 아니고, 일부는 따로 제작된 프로그램을 통해 자동으로 수행하였고, 또 다른 일부는 별도로 학습된 대형 neural network(e.g. ResNet-151)을 사용하여 얻어진 semantic label을 그대로 사용하였습니다.

앞서 언급한 26개 task들을 아래와 같이 나타내었습니다. 이들 각각에 대한 구체적인 설명은, 해당 논문의 <a href="http://taskonomy.stanford.edu/taskonomy_supp_CVPR2018.pdf" target="_blank">보충 자료</a>를 참조해 주시길 바랍니다. 

{% include table.html description="논문에서 채택된 26가지 task 리스트" content="
| Autoencoding | Colorization | Context Encoding |
| Context Prediction (Jigsaw) | Curvature Estimation | Denoising |
| Depth Estimation, Euclidean | Depth Estimation, Z-Buffer | Edge Detection (2D) |
| Edge Detection (3D) | Keypoint Detection (2D) | Keypoint Detection (3D) |
| Point Matching | Relative Camera Pose Estimation, Non-Fixated | Relative Camera Pose Estimation, Fixated |
| Relative Camera Pose Estimation, Triplets (Egomotion) | Reshading | Room Layout Estimation |
| Segmentation, Unsupervised (2D) | Segmentation, Unsupervised (2.5D) | Surface Normal Estimation |
| Vanishing Point Estimation | Semantic Learning through Knowledge Distillation: Classification, Semantic (1000-classes) | Semantic Learning through Knowledge Distillation: Classification, Semantic (1000-classes) |
| Semantic Learning through Knowledge Distillation: Classification, Semantic (100-classes) | Semantic Learning through Knowledge Distillation: Segmentation, Semantic | |
" class="full-table" %}

{% include image.html name=page.name file="task-dictionary-examples.png" description="논문에서 채택된 26가지 중 24가지 task 예측 결과 예시" class="large-image" %}

### 3.1. Step I: Task-Specific Modeling

맨 먼저 $$\mathcal{S}$$ 내의 각 task $$s_i$$에 대하여 task-specific networks를 독립적으로 학습합니다. 각 task-specific network는 공통적으로 아래와 같은 encoder-decoder 구조를 지닙니다. 이 때, decoder의 경우 task의 목적에 따라 출력값을 생성하는 부분의 구조에 약간의 차이가 있습니다.

{% include image.html name=page.name file="task-specific-network-architecture.png" description="Task-Specific network의 encoder-decoder 구조" class="large-image" %}

### 3.2. Step II: Transfer Modeling

$$s \in \mathcal{S}$$와 $$t \in \mathcal{T}$$인 어느 source task $$s$$와 target task $$t$$에 대하여, $$s$$의 task-specific network의 encoder $$E_s(\cdot)$$와, parameters $$\theta$$로 표현되는 새로운 decoder $$D_{\theta}(\cdot)$$이 합쳐져 구성된 transfer network를 생성합니다. 그러면, 어느 입력 이미지 $$I$$에 대한 transfer network의 예측 레이블은 $$D_{\theta}(E_s(I))$$로 표현할 수 있습니다.

{% include image.html name=page.name file="transfer-network-architecture.png" description="Transfer network의 구조" class="large-image" %}

입력 이미지 $$I$$에 대한 target task $$t$$의 ground truth 레이블을 $$f_t(I)$$로 표현하고, 해당 task의 loss function(손실 함수)을 $$L_t$$로 표현한다면, 전체 training set $$\mathcal{D}$$에 대한 loss를 최소화하는 $$\theta$$는 아래의 minimization 수식으로 표현할 수 있습니다.

\begin{equation}
D\_{s \to t} := \arg\min\_{\theta} \mathbb{E}\_{I \in \mathcal{D}} \big[ L_t \big( D\_{\theta} ( E_s(I)), f_t(I) \big) \big]
\end{equation}

위 수식의 계산 결과 얻어진 최적의 transfer function $$D_{s \to t}$$를 *readout function*이라고도 하며, $$D_{s \to t}$$의 성능이 우수할수록 두 task $$s$$, $$t$$ 간의 transferability가 높다고 해석할 수 있습니다. 모든 $$(s, t)$$ 조합에 대한 readout function들을 모두 구합니다.

한편 위에서 서술한 transfer modeling의 과정은, 엄밀하게는 transfer order $$k=1$$인 경우에 해당하였습니다. $$k$$가 1보다 큰 경우, 즉 source task들이 2개 이상 사용될 수 있는 경우에는, 모든 target task 경우에 대하여 전체 조합 경우의 수가 총 $$\vert \mathcal{T} \vert \times { {\vert \mathcal{S} \vert }\choose{k}}$$개가 됩니다. $$k=2$$인 경우만을 생각하더라도, $$\vert \mathcal{T} \vert=22$$, $$\vert \mathcal{S} \vert=25$$일 때 총 $$(22 \times { {25}\choose{2}})=6,600$$개가 됩니다. 

지나치게 많은 계산을 방지하고자, Taskonomy 방법에서는 먼저 $$k=1$$로 하여 계산한 모든 $$D_{s \to t}$$의 성능을 기준으로 hypergraph를 그리고, 여기에 <a href="https://en.wikipedia.org/wiki/Beam_search" target="_blank">beam search</a>를 적용하여 $$D_{s \to t}$$ 성능 기준으로 상위에 속하는 5개($$k \leq 5$$인 경우) 또는 $$k$$개($$k \geq 5$$인 경우)의 source task들을 취사 선택하고, 이들 간의 $$k$$차 조합만을 고려하는 방식을 채택합니다. 이렇게 하여, 모든 $$({s_1, ..., s_k}, t)$$ 조합에 대한 readout function들을 모두 구합니다. 

{% include image.html name=page.name file="beam-search-keq2-examples.png" description="Transfer order k=2인 경우의 beam search 과정 예시" class="full-image" %}

### 3.3. Step III: Ordinal Normalization using Analytic Hierarchy Process (AHP)

이제 각 task들 간의 readout function들을 사용, 이들 간의 transferability를 기반으로 affinity matrix를 계산해야 하는데, 이 때 문제가 하나 있습니다. 실제로 transferability를 정량화하여 affinity matrix를 계산해내기 위한 지표를 설정해야 하는데, 가장 쉽게 생각할 수 있는 것이 readout function의 loss $$L_{s \to t}$$일 것입니다. 그런데, 실제 target task $$t$$에 따라 $$L_{s \to t}$$ 값의 범위가 천차만별이고, $$L_{s \to t}$$ 값의 감소에 따른 실제 체감되는 예측 결과 품질의 증가 속도도 task 별로 차이가 존재합니다. 이로 인해 $$L_{s \to t}$$를 단순히 normalize하는 것만으로는 모든 target task들을 동일 선상에서 커버하기에 충분하지 않으므로, Taskonomy 방법에서는 대신 **ordinal(서수)**에 기반한 normalization을 수행합니다.

이해를 돕기 위해 transfer order $$k=1$$인 경우를 기준으로 진행하겠습니다. 어느 target task $$t$$에 대하여, $$i$$번째 source task $$s_i$$과 $$j$$번째 source task $$s_j$$를 각각 적용하여 얻어진 readout function들을 별도의 test set $$\mathcal{D}_{test}$$을 사용하여 테스트한 결과, $$s_i$$로부터의 예측 성능이 $$s_j$$로부터의 예측 성능보다 우수하였던(즉, $$\mathcal{D}_{s_i \to t}(I) > \mathcal{D}_{s_j \to t}(I)$$) *테스트 이미지($$I$$)의 수*를 카운팅하고, test set 내에서의 그 비율을 계산할 수 있습니다. $$i$$와 $$j$$를 변경해 가면서 해당 비율을 계산하여 이를 pairwise matrix $$W_t$$로 나타낸다고 하면, $$W_t$$의 $$(i,j)$$번째 성분 $$w_{i,j}$$는 아래와 같이 표현됩니다:

\begin{equation}
w_{i,j} = \mathbb{E}\_{I \in \mathcal{D}\_{test}} [\mathcal{D}\_{s_i \to t}(I) > \mathcal{D}\_{s_j \to t}(I)]
\end{equation}

그런 뒤 $$W_t$$를 $$[0.001, 0.999]$$ 범위 안으로 clipping하고, 새로운 matrix $$W_t' = W_t / W_t^T$$ (element-wise division)를 계산합니다. 이를 통해 $$s_i$$가 $$s_j$$에 비해 '평균적으로 성능이 몇 회나 더 우수하였는지'를 $$W_t'$$이 포함하고 있도록 합니다. 즉, $$W_t'$$의 $$(i,j)$$번째 성분 $$w_{i,j}'$$는 아래와 같이 표현됩니다:

\begin{equation}
w_{i,j}' = \frac { \mathbb{E}\_{I \in \mathcal{D}\_{test}} [\mathcal{D}\_{s_i \to t}(I) > \mathcal{D}\_{s_j \to t}(I)] } { \mathbb{E}\_{I \in \mathcal{D}\_{test}} [\mathcal{D}\_{s_i \to t}(I) < \mathcal{D}\_{s_j \to t}(I)] }
\end{equation}

이렇게 얻어진 $$W_t'$$의 principal eigenvector(eigenvalue가 가장 큰 eigenvector)를 계산합니다. 그러면 해당 principal eigenvector의 $$i$$번째 성분은, 곧 이에 대응되는 $$i$$번째 source task $$s_i$$의, source task들로 구성한 undirected graph 상에서의 <a href="https://en.wikipedia.org/wiki/Eigenvector_centrality" target="_blank">centrality(중심성, 구심성)</a>를 나타내게 됩니다. 이는 다른 source task들 대비, 해당 source task의 target task에 대한 일종의 '영향력'이라고 봐도 크게 무리가 없겠습니다. 저자들은 이러한 normalization 방법을, operations research(경영과학) 등의 분야에서 흔히 사용되는 <a href="https://en.wikipedia.org/wiki/Analytic_hierarchy_process" target="_blank">Analytic Hierarchy Process(AHP)</a>로부터 채택하였다고 언급하고 있습니다.

모든 $$t \in \mathcal{T}$$에 대하여 $$W_t'$$의 principal eigenvector들을 계산하여 이들을 row-wise로 쌓아올리면, 아래 그림과 같은 task affinity matrix $$P$$가 얻어집니다.

{% include image.html name=page.name file="task-affinity-matrix-1st-order-example.png" description="k=1일 때의 task affinity matrix 예시<br><small>(여기에서는 값이 작을수록, source->target transferability가 높음)</small>" class="full-image" %}

논문 본문에서는 $$k=1$$인 경우만을 가지고 설명하였는데, $$k$$가 $$1$$보다 큰 경우에는 task affinity matrix를 계산하는 과정에서의 성능 비교 대상이, 엄밀히 말하면 source task들이 기준이 되는 것이 아니라, $$k$$차 조합으로 구성한 $$(s_1, ..., s_k, t)$$ 조합들이 기준이 됩니다. ($$k=1$$인 경우 $$(s_k, t)$$ 조합 안에 하나의 source task만이 포함되기 때문에 이를 source task 자체로 대표하여 표현할 수 있었던 것이라고 볼 수 있습니다.) 즉 다시 말해, task affinity matrix $$P$$ 상의 하나의 열은, 곧 하나의 $$(s_1, ..., s_k, t)$$ 조합을 대표한다고 이해하시면 되겠습니다. 이해를 돕기 위해, $$k=1$$과 $$k=2$$인 경우 $$P$$의 각 열을 설명하는 그림을 아래에 추가하였습니다.

{% include image.html name=page.name file="task-affinity-matrix-higher-order-example.png" description="k >= 1일 때의 task affinity matrix 예시<br><small>(여기에서는 값이 작을수록, source->target transferability가 높음)</small>" class="full-image" %}


### 3.4. Step IV: Computing the Global Taxonomy

Task affinity matrix $$P$$가 완성되면, 이를 사용하여 특정한 target task의 성능을 극대화하는 최적의 transfer policy를 탐색하는 작업을 마지막으로 진행합니다. 즉, target task의 성능을 극대화하도록 하는 $$(s_1, ..., s_k, t)$$ 조합을 선택하는데, 이 과정에서 선택된 source task들의 갯수가 처음에 상정했던 supervision budget $$\gamma$$를 초과하지 않도록 제약을 걸어야 합니다.

논문 본문에서는 이 지점부터 notation이 약간 바뀌어서 혼동될 수 있는데, 항상 기준은 $$(s_1, ..., s_k, t)$$ 조합으로 보면 됩니다. $$(s_1, ..., s_k)$$와 $$t$$를 연결하는 transfer(edge)를 기본 단위로 하여 최적의 transfer를 탐색합니다. $$i$$는 transfer, $$j$$는 target task의 index를 나타냅니다. 또한, $$sources(i)$$는 $$i$$번째 transfer의 source task들의 set을 지칭하고, $$target(i)$$는 $$i$$번째 transfer의 target task 하나를 지칭합니다.

앞서 찾은 $$P$$와, transfer 규칙 및 supervision budget과 관련된 제약 조건을 적용하여, <a href="https://en.wikipedia.org/wiki/Integer_programming" target="_blank">Boolean Integer Programming(이하 BIP)</a> 문제의 objective function(목적 함수) 및 constraints(제약식)를 설정합니다. 이 때 boolean variables $$x = (x_1, x_2, ..., x_{\vert E \vert+ \vert \mathcal{V} \vert})$$은, 전체 $$E$$개의 transfer(edge)들과 $$\mathcal{V}$$개의 source task들 중에서 어느 것들을 선택할지를 $$\{0,1\}$$ 중 하나로 표시합니다. 결과적으로, 아래와 같은 BIP 문제를 풀면 됩니다:

{% include image.html name=page.name file="bip-objective-and-constraints.png" class="small-image" %}

사실 이 대목에서 BIP를 이해하고자 너무 머리 싸매고(?) 노력하실 필요는 없을 것 같습니다. 중요한 것은 실제 Taskonomy hypergraph에서 고려하고자 하는 조건에 맞게 objective function의 weights와 더불어 constraints를 적절하게 입력하는 것이고, 일단 이것에 성공하면 실제 BIP 문제는 <a href="http://www.gurobi.com" target="_blank">Gurobi Optimizer</a> 등의 최적화 문제 풀이용 프로그램이 알아서 풀어줍니다. 단, 논문 본문에 나와 있는 weights 및 constraints에 대한 설명이 다소 혼동의 여지가 있어, 결과적으로 각각을 어떻게 입력하면 되는지 위주로 좀 더 자세히 서술해 보았습니다. 

먼저 objective function weights의 경우, 전체 $$\vert E \vert+\vert \mathcal{V} \vert$$개의 원소 중 앞쪽 $$\vert E \vert$$개의 transfer(edge)들에 대응되는 것만 $$c_i := r_{target(i)} \cdot p_i$$를 넣어주고, 나머지 원소에는 $$0$$을 넣어주면 됩니다. 이 때, $$p_i$$는 $$i$$번째 transfer가 가리키는 target task에 대한 transferability 수준을 나타내며, 이는 앞서 계산하여 얻은 task affinity matrix $$P$$ 상에서 찾을 수 있습니다. 다음으로 $$r_{target(i)}$$는 $$i$$번째 transfer가 가리키는 target task의 상대적 중요성을 나타내며, 이는 필요에 따라 사용자가 적절하게 정하여 입력할 수 있습니다.

{% include image.html name=page.name file="bip-weights-description.png" description="BIP 문제에서의 objective weights 값의 설정" class="large-image" %}

다음으로 constraints에는 크게 다음 3가지 조건을 반영합니다. 

1. 만약 어느 subgraph 상에 특정 transfer가 포함되어 있다면, 해당 transfer의 source task(node)들 또한 반드시 포함되어야 한다.
2. 각 target task로 반드시 딱 하나의 transfer가 들어간다.
3. 전체 supervision budget $$\gamma$$를 초과하지 않도록, source task들을 선정해야 한다.

위 3가지 조건을 모두 반영하면 $$A \in \mathbb{R}^{(\vert E \vert + \vert \mathcal{V} \vert + 1) \times (\vert E \vert + \vert \mathcal{V} \vert)}$$, $$b \in \mathbb{R}^{(\vert E \vert + \vert \mathcal{V} \vert + 1)}$$를 얻을 수 있으며, 좀 더 구체적으로는 $$A$$와 $$b$$ 내 각 구역 별 원소의 값들을 아래 그림에 표시한 조건에 따라 결정하면 됩니다. 이 때, $$l_i$$는 $$i$$번째 transfer와 결부된 source task들을 레이블링할 시의 cost를 가리킵니다.

{% include image.html name=page.name file="bip-constraints-description.png" description="BIP 문제에서의 constraints 설정<br><small>(클릭하면 확대하여 보실 수 있습니다)</small>" class="large-image" %}


## 4. Experiments

논문에서는 총 26개 task들 중 4개의 task들(e.g. colorization, jigsaw puzzle, in-painting, random projection)을 source-only task로 정의하고, 이들을 제외한 나머지 22개 task들 중 하나로 target task로 선정하고, transfer order $$k$$를 1부터 25까지 증가시켜 나가면서 Taskonomy 방법에 대한 실험을 반복했습니다. 전체 경우의 수로부터 발생하는 transfer function들의 수는 약 3,000개에 육박하였으며, 총 47,886 GPU hours가 소요되었다고 하였습니다.

모든 실험에서 task-specific network의 encoder는 ResNet-50(pooling layer 제외) 구조를 채택하였고, transfer network의 decoder는 target task의 종류에 따라 그 구조를 조금씩 다르게 설계했습니다. 실험에 사용한 데이터셋은 400만 장의 원본 이미지 중 일부를 랜덤 샘플링하여 training set(12만), validation set(1.6만), test set(1.7만)으로 분할하고, task-specific network는 training set으로, transfer network는 validation set으로 학습하였습니다.

본 논문에서의 테스트 결과 성능은 *win rate(%)*이라는 지표를 주로 사용하여 표현하였습니다. 이는 Taskonomy 방법과 비교하고자 하는 baseline 방법이 있을 때, 전체 test set 중에서 Taskonomy 방법의 예측 성능이 baseline 방법의 예측 성능보다 우수하였던 이미지 수 비율을 나타냅니다. 

본격적인 Tasknomy 학습 결과 분석에 앞서, 학습이 완료된 task-specific network의 성능이 기본적으로 쓸 만한지(?)를 먼저 간단히 검증하였습니다. 정규분포로부터 샘플링한 랜덤한 값들을 그대로 task-specific network의 weight들로 사용하여 테스트하는 방법(*rand*)와, 각 task 별 실제 레이블들을 평균한 결과를 사용하여 테스트하는 통계적인 방법(*avg*)을 baseline으로 하여 검증한 결과, 충분히 안정적으로 잘 학습되었다는 것을 확인하였습니다. 아래 그림에서 그 결과를 확인할 수 있습니다.

{% include image.html name=page.name file="task-specific-networks-sanity.png" description="Task-specific network들의 각 task 별 성능 검증 결과" class="large-image" %}

### 4.1. Evaluation of Computed Taxonomies

Supervision budget $$\gamma$$ 및 Transfer Order $$k$$를 변경해 가면서 학습한 결과 얻어진 몇 가지 예시 Taskonomy들을 아래 그림과 같이 나타냈습니다. 

{% include image.html name=page.name file="computed-taxonomies.png" description="학습을 통해 얻어진 예시 Taskonomy<br><small>(BIP 문제를 해결하여 얻어진 transfer들을 화살표로 표현함; 흐릿하게 표시된 node들은 source-only task에 해당함)</small><br><small>(클릭하면 확대하여 보실 수 있습니다)</small>" class="full-image" %}

그림의 우측에 확대하여 나타낸 $$\gamma=8$$, $$k=4$$인 경우를 예로 들어 좀 더 자세히 살펴봅시다. 이미지 상의 물체들의 (각 방향으로의) 표면들을 검출하는 Surface Normal Estimation(*'Normals'*) task가, source task들 중 하나로써 다른 다양한 target task에 대하여 transfer되고 있는 것을 확인할 수 있습니다. 이는 이미지 상에 보여진 공간에 대한 이해를 수행하는 데 있어, 물체들의 표면을 검출하는 작업이 중대한 영향을 미칠 수 있다는 사실의 간접적인 증거로써 확인할 수 있었습니다. 한편 컴퓨터 비전 분야에서 전통적으로 연구되어 오던, 이미지 상의 특징적인 부분을 검출하는 Keypoint Detection('*2D Keypoints*')의 경우, Denoising, Colorization, In-painting 등의 Unsupervised Learning task들로부터의 transfer를 통해 도움을 얻을 수 있다는 흥미로운 결론도 확인할 수 있었습니다.

다음으로 좀 더 정량적인 결과 확인을 위해, 완성된 taskonomy에 기반하여 얻어진 transfer 규칙들을 각 target task에 적용하여 transfer learning을 수행했을 시의 성능 결과를 조사하였습니다. 이 때, 'Gain'과 'Quality'라는 두 가지 지표를 사용하였습니다.

- *Gain*: Transfer network의 학습에 사용한 validation set(1.6만)으로, target task의 task-specific network를 처음부터 학습하는 방법을 baseline으로 설정했을 시의, taskonomy 방법의 win rate(%)
- *Quality*: Task-specific network의 학습에 사용한 training set(12만)으로,  target task의 task-specific network를 처음부터 학습하는 방법을 baseline으로 설정했을 시의, taskonomy 방법의 win rate(%)

{% include image.html name=page.name file="taxonomy-evaluation.png" description="완성된 taskonomy에 기반한 transfer 규칙을 각 target task에 적용하였을 시의 테스트 결과<br><small>(클릭하면 확대하여 보실 수 있습니다)</small>" class="large-image" %}

일단 Maximum transfer order $$k$$를 증가시킬수록, 그리고 supervision budget $$\gamma$$를 증가시킬수록, Gain과 Quality가 점차적으로 증가하는 경향을 보였습니다. 이는 '더 많은 source task로부터 얻은 지식을 transfer할 수록 성능이 높아질 것이다'라는 우리의 상식적인 예상대로 전개된 결과라고 할 수 있습니다. 

한편 *Quality* 지표에서 상정한 baseline의 경우 약 10배 더 많은 양의 학습 데이터를 사용했기 때문에, Quality의 절대적인 수치가 상대적으로 저조하게 나오는 것은 어느 정도 당연한 결과라고 할 수 있습니다. 그럼에도 불구하고, maximum $$k$$ 또는 $$\gamma$$가 증가했을 때 대부분의 task에서의 Quality 값이 0.5에 도달했다는 것은, baseline 방법에 비해 학습 데이터의 엄청난 양적 열세가 있었음에도 불구하고 그 성능이 baseline 방법의 그것과 거의 비등하였다는 사실을 보여주며, 이는 taskonomy 방법을 통해 찾은 '효과적인' transfer policy를 적용했을 시의 위력(?)을 아주 잘 보여주는 대목이라고 할 수 있겠습니다. 

### 4.2. Generalization to Novel Tasks

앞선 실험에서는 사전에 가정한 어느 task dictionary 내의 모든 source-target 조합들을 고려하여 taskonomy를 계산하는 과정을 거쳤습니다. 그런데 좀 더 현실적인 상황에서는 이전에 고려되지 않았던 어느 새로운 target task에 대하여, 기존에 고려하고 있던 source task들을 사용한 최적의 transfer policy를 찾는 것에 좀 더 관심이 많을 것입니다.

이러한 상황을 재현하고자, 기존 task dictionary 내의 모든 task들을 source로, 기존 task dictionary에 없었던 어느 새로운 task를 단일 target으로 한 일반화 성능 검증 실험을 수행하였습니다. 이 때 target task와 관련된 학습 데이터는 validation set(1.6만)만을 사용하였음을 다시 강조합니다. 

{% include image.html name=page.name file="generalization-to-novel-tasks.png" description="새로운 task에 대한 일반화 성능 검증 결과<br><small>(좌측: Gain과 Quality; 우측: 상기 명시된 방법들을 baseline으로 한 win rate(%))</small><br><small>(클릭하면 확대하여 보실 수 있습니다)</small>" class="large-image" %}

위 실험 결과 중 특히 우측에 나타낸, 최신 transfer learning 관련 방법들과의 비교 결과가 인상적입니다. 각종 self-supervised 방법들을 사용한 경우, ImageNet 데이터셋으로 학습한 AlexNet의 FC7을 features로 사용한 경우 등에 비해, 완성된 taskonomy에 기반하여 찾은 transfer policy에 따라 학습한 경우의 성능이 전체적으로 더 우수한 것으로 나타났습니다.


## 5. Significance Test of the Structure

위의 과정을 거쳐 찾은 Taskonomy 결과에 대하여, 본 논문에서는 다양한 방법으로 검증을 시도하였습니다.

{% include image.html name=page.name file="structure-significance.png" description="랜덤한 transfer policy 대비 Taskonomy 방법으로 찾은 최적의 transfer policy 적용 성능 비교 결과<br><small>(녹색: Taskonomy; 회색: 랜덤 transfer policy)</small>" class="large-image" %}

Taskonomy 방법을 통해 찾은 최적의 transfer policy가, 랜덤하게 정의된 transfer policy에 비해 얼마나 더 효과가 있는지 확인하기 위해, 간단한 유의성(significance) 검증 실험을 수행하였습니다. 모든 supervision budget $$\gamma$$에서 Taskonomy 방법을 통해 찾은 transfer policy의 성능이 Quality와 Gain 두 가지 지표에서 월등하게 우수한 것으로 확인되었습니다. 이는 곧 서로 다른 task들 간에 모종의 structure가 존재한다는 것을 방증하며, 이를 Taskonomy 방법이 잘 모델링하였다는 것을 보여줍니다.

### 5.1. Evaluation on MIT Places & ImageNet

데이터셋 차원에서의 간단한 일반화 성능 검증을 위해, Object Classification task로 ImageNet, Scene Classification task로 MIT Places를 각각 target task의 데이터셋으로 선정하고, 여기에 대하여 각 source task들의 task-specific network들을 fine-tuning한 결과 테스트 성능을 조사하였습니다. 이 성능을 기준으로 한 랭킹 결과를 산출하고, 이를 앞서 taskonomy 방법을 통해 찾은 transferability를 기준으로 한 랭킹 결과와 비교하여 서로 간에 상관성이 존재하는지 알아보고자 하였습니다.

{% include image.html name=page.name file="transferability-correlations.png" description="ImageNet, MIT Places 데이터셋으로의 transferability 결과와의 상관성 조사 결과<br><small>(클릭하면 확대하여 보실 수 있습니다)</small>" class="large-image" %}

<a href="https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient" target="_blank">Spearman's rho</a>를 기준으로, MIT Places와 $$0.857$$, ImageNet과 $$0.823$$을 나타낸 것으로 볼 때, 유의미한 수준의 상관성이 존재함을 확인하였습니다.

### 5.2. Universality of the Structure

Taskonomy 방법의 안정성 검증을 위해, 아래의 요소들을 변화시켜 가면서 추가적인 실험을 수행하였습니다.

- Task-specific network의 구조
- Transfer network의 구조
- Transfer network 학습 시 사용 가능한 학습 데이터의 양
- 실험에 사용한 데이터셋
- Training/Validation/Test set 분할 결과물
- Task dictionary

변화의 범위를 적지 않은 수준으로 주었음에도 불구하고, 위에서 조사된 Taskonomy 결과에서 크게 벗어나지 않는 결과물을 얻었다고 보고하였습니다(자세한 내용은 <a href="http://taskonomy.stanford.edu/taskonomy_supp_CVPR2018.pdf" target="_blank">보충 자료</a>을 통해 확인할 수 있습니다).

### 5.3. Task Similarity Tree

각 task들 간의 structure를 좀 더 명시적으로 조사하기 위해, 위에서 구한 task affinity matrix $$P$$를 사용한 <a href="https://en.wikipedia.org/wiki/Hierarchical_clustering" target="_blank">병합 군집(agglomerative clustering)</a>을 통해 서로 다른 task 간의 유사도를 조사하였습니다. 

{% include image.html name=page.name file="task-similarity-tree.png" description="Task 유사도 트리" class="large-image" %}

군집화 결과 트리 형태의 task 계층도가 얻어진 것을 확인할 수 있습니다. 크게 3D task(e.g. Surface Normals, 2.5D Segmentation 등), 2D task(e.g. 2D Edges, Autoencoding 등), 저차원 상의 기하학적 task(e.g. Room Layout, Vanishing Points 등), 그리고 의미적 task(e.g. Semantic Segmentation, Object Classification 등)가 각각 군집으로 묶인 것을 확인할 수 있으며, 이는 우리의 직관에서 크게 벗어나지 않는 결과라고 할 수 있습니다.


## 6. Limitations and Discussion

본 논문에서는 복수 개의 서로 다른 task들 간에 잠재적으로 존재하는 관계들을 모델링하는 Taskonomy 방법을 통해, 새로운 task에 대한 딥러닝 모델의 학습을 보다 효과적이고 효율적으로 수행할 수 있다는 것을 보였습니다. 이는 task들이 구성하는 모종의 공간인 task space가 존재함을 가정하고,  이를 규명하기 위한 최초의 시도를 했다고 할 수 있겠습니다. 다만, 저자들은 본 Taskonomy 방법을 구상할 때 몇 가지 가정이 들어갔기 때문에, 이들을 점차적으로 완화하는 것이 곧 향후 연구를 통해 추구해야 할 방향임을 역설하면서 내용을 마무리하였습니다.

(1) Model Dependence: 본 논문에서는 학습 모델을 deep neural networks로, 데이터셋을 이미지 데이터셋으로 한정하였기 때문에, 실험 결과가 model-specific하면서 동시에 data-specific하다는 점을 지적하였습니다.

(2) Compositionality: 본 논문에서 다룬 task들은 모두 사람이 정의한 task에 해당합니다. 만일 이들 task가 적절하게 조합될 경우, 이를 통해 모종의 새로운 subtask들을 발견할 수 있을지에 대한 의문을 제기하였습니다.

(3) Space Regularity: 본 논문에서는 앞서 가정한 모종의 task space에서 샘플링을 통해 얻은 task dictionary를 사용한 결과만을 도출하였습니다. 이보다 좀 더 일반적인 task 샘플링 결과에 대해서도 그 효과가 검증될 수 있을지, 즉 '정규성'에 대한 검증이 추가로 필요함을 지적하였습니다.

(4) Transferring to Non-visual and Robotic Tasks: 본 논문에서는 모두 이미지와 관련된 visual task들에 대해서만 검증을 수행하였습니다. 자연히, 로봇 조작과 같이 완전히 시각적이지 않은 분야에서도 Taskonomy 방법을 통해 transferability를 극대화할 수 있을지에 대한 의문을 제기하였습니다.

(5) Lifelong Learning: 본 논문에서는 Taskonomy를 완성하는 작업을 단 한 번에 수행하였습니다. 다만 오늘날에는 어느 시스템 자체가 계속적인 학습을 수행하면서 그것이 수행할 수 있는 task를 점진적으로 확장시킬 수 있을지에 관심이 집중되고 있으며, 그것이 'lifelong learning'이라는 이름의 연구 주제로 진행되고 있습니다. 이에 따라 lifelong learning 셋팅에서의 Taskonomy 방법에 대한 검증을 고려해봐야 함을 지적하였습니다.


## References

- Taskonomy 논문
  - <a href="https://arxiv.org/abs/1804.08328" target="_blank">Zamir, Amir R., et al. "Taskonomy: Disentangling Task Transfer Learning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.</a>
- Taskonomy 논문: 보충 자료 
  - <a href="http://taskonomy.stanford.edu/taskonomy_supp_CVPR2018.pdf" target="_blank">Zamir, Amir R., et al. "Taskonomy: Disentangling Task Transfer Learning - Supplementary Material." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.</a>
