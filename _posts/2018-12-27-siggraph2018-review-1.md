---
layout: post
title: "SIGGRAPH 2018 리뷰: (1) 참석 후기 및 프로그램 소개"
date: 2018-12-27 14:00:00 +0900
author: hoseong_lee
categories: [Review]
tags: [siggraph2018, computer-graphics]
comments: true
name: siggraph2018-review-1
redirect_from: "/machine-learning/computer-vision/2018/12/27/siggraph2018-review-1.html"
---

안녕하세요, 다소 늦은 감이 있지만 지난 8월 캐나다 밴쿠버에서 개최되었던 SIGGRAPH 2018 학회의 참석 후기와, 주요 프로그램에 대한 소개, 그리고 제가 재미있게 들었던 논문 한편에 대한 리뷰 및 4편에 대한 간단한 소개를 다룰 예정입니다. 
이번 주제는 두 편에 걸쳐서 작성할 예정이며, 본 편에서는 SIGGRAPH 참석 후기와 프로그램에 대한 소개를 다룰 예정입니다. 

SIGGRAPH 2018은 수아랩의 지원을 받아서 참석하였으며, 저처럼 그래픽스 전공자가 아닌 사람들도 즐길 수 있었던 좋은 학회였습니다. 
무엇보다 덥고 습한 한국과는 다르게 여름이지만 선선하고 쾌적한 밴쿠버의 기후와 분위기가 가장 기억에 남았습니다. 
학회장에서 느낀 점들과 어떤 프로그램들이 있는지에 대해 소개를 드리도록 하겠습니다.

## SIGGRAPH

### What is SIGGRAPH?

{% include image.html name=page.name file="fig1.PNG" description="" class="full-image" %}

SIGGRAPH는 Special Interest Group on Graphics and Interactive Techniques 라는 Full name을 가지고 있는 학회이며 1974년에 시작해서 올해로 45회차를 맞은 역사가 깊은 학회입니다. 
저처럼 컴퓨터 비젼, 그리고 머신러닝과 딥러닝을 연구하는 사람들에겐 다소 생소할 수 있지만 **컴퓨터 그래픽스 분야** 에서는 최대 규모를 자랑하는 학회로 알려져 있습니다. 
또한 올해로 11회를 맞이한 SIGGRAPH Asia처럼 아시아를 타겟으로 같은 학회가 같은 해에 한 번 더 열리기도 합니다. (올해는 얼마전 도쿄에서 열렸다고 합니다.) 

{% include image.html name=page.name file="fig2.png" description="" class="full-image" %}

사진에 보이는 장소가 바로 올해 SIGGRAPH 2018이 개최된 장소인 Vancouver Convention Centre입니다. 
학회에서 발표를 듣고 잠시 나오면 바로 바다가 보여서 정말 평화롭고 운치 있었던 것 같습니다. 

### SIGGRAPH Paper Acceptance rate 

최근 5년간 SIGGRAPH의 논문 accepted rate는 **평균 25%** 정도이며 매년 약 450~ 500편 전후의 논문이 submitted 되고 그 중 약 120편 정도의 논문이 accepted 되는 규모를 보이고 있습니다. 
적당한 규모의 학회이며 실제로 학회에서는 총 38개의 주제로 나뉘어서 발표가 진행이 되었습니다. 
SIGGRAPH 2018에 발표된 논문의 리스트는 
<a href="https://s2018.siggraph.org/conference/conference-overview/technical-papers/" target="_blank"><b> 다음 링크 </b></a>
에서 확인이 가능합니다. 


## 주요 프로그램 소개

### - Exhibition
SIGGRAPH는 논문 발표 외에도 각종 그래픽스 관련 다양한 주제의 데모와 포스터, 전시 등이 활발하여 단순히 논문 발표만 듣는 것이 아니라, 데모와 전시 등의 “보는 재미”도 느낄 수 있는 학회였습니다. 

{% include image.html name=page.name file="fig3.PNG" description="" class="full-image" %}

위 사진은 제가 전시장을 돌아다니며 찍은 사진들입니다. 
요즘 뉴스에서 자주 보이는 VR, AR 등의 가상현실, 증강현실 데모가 주를 이뤘고, 사진에서 보이시는 것처럼 가상현실 체험이 인기가 많았습니다. 
또한 단순 Technology를 넘어서 Art에도 그래픽스 연구들을 접목시킨 데모들이 많았고 우측 하단 사진은 SIGGRAPH를 뜨겁게 달궜던 NVIDIA의 GPU를 이용한 실시간 Style Transfer 데모를 체험하고 있는 것을 담아보았습니다. 
약 10초마다 Source가 되는 Style(우측 하단)이 변하면 실시간으로 Target이 되는 제 모습과 배경도 변하는 것이 인상 깊었으며, 많은 연산량을 필요로 하는 task인데 본인들의 1000만원이 넘는 GPU를 사용하면 실시간이 가능하다! 고 설명을 하는 것이 기억에 남네요. 

또한 VR Theater, Electronic Theater라고 해서 학회 중에 영화를 관람할 수 있는 프로그램이 있는데, 티켓을 구입하거나 등록할 때 추가 요금을 지불하면 짧은 단편 영화 여러 편을 감상할 수 있습니다. 
개인적으로는 Electronic Theater도 기억에 남았던 것 같습니다. 
단순 단편 영화가 아니라, 최신 컴퓨터 그래픽스 기술, 흔히들 CG라 부르는 기술들이 접목되어 있는 단편 애니메이션 영화들이 주를 이루었고 내용도 재미있어서 시간 가는 줄 모르고 봤습니다. 
티저 영상은 
<a href="https://www.youtube.com/watch?v=Kq1sOZChpwI" target="_blank"><b> 해당 링크 </b></a>
에서 확인이 가능합니다. 

### - Training Session
다음으로 설명드릴 프로그램은 Training Session입니다. 
다양한 기업에서 여러 주제로 Course를 진행하는 프로그램으로, 저는 주로 NVIDIA에서 진행한 “Hands-on training session”에 참여를 하였습니다. 
각 강의 당 1시간 30분 ~ 2시간 정도 진행이 되며 딥러닝을 이용하여 이미지를 다루는 여러 주제들의 튜토리얼로 진행이 되었습니다. 
제가 들은 주제들은 다음과 같습니다.

-	“Image Super Resolution using Autoencoders”
-	“Analogous Image generation using CycleGAN”
-	“Image creation using Generative Adversarial Networks in Tensorflow and DIGITS”
-	“Anomaly Detection with Variational Autoencoder”
-	“Image Style Transfer with Torch”

이 외에도 강화학습, Character Animation 등 다양한 주제들이 있었지만 저는 주로 이미지 관련 Training을 들었습니다. 
강의는 정해진 수업 시간에만 사용 가능한 aws cloud가 제공이 되어서 강의실에서 cloud에 접속을 하고 실습 코드(jupyter notebook)를 강사와 같이 돌려보면서 질의 응답을 하는 식으로 진행이 되었습니다. 
저는 주로 TensorFlow를 사용하지만 최근 PyTorch도 공부를 하고 있어서 코드를 이해하며 공부를 할 수 있었습니다. 
이 외에도 Unity 등 다른 기업에서도 다양한 주제로 Training Session을 열고 있으므로, 학회장에서 직접 코드를 돌려보고 현업 기술자들과 소통을 할 수 있는 좋은 기회라고 생각합니다.

### - Technical Paper Fast Forward
다음 설명드릴 프로그램은 학회의 꽃인 논문과 관련된 프로그램입니다. 
100 ~ 200편의 논문이 매년 accept되기도 하고 무엇보다 다양한 주제로 논문이 발표가 되니까 어떤 논문을 읽고 발표를 들을 지 굉장히 감을 잡기가 어려웠습니다. 
그런데 마침 **“Fast Forward”** 라는 프로그램이 이 고민을 해결해주었습니다. 
학회가 시작된 첫날 저녁에 열리는 프로그램인데, 저와 같은 고민을 하는 사람들을 위해 모든 구두 발표자들이 한 장소에 모여서 더도 말고 덜도 말고 딱 **30초** 간 본인의 연구를 소개합니다. 
엄격하게 시간을 준수하는 것이 인상깊었습니다. 

{% include image.html name=page.name file="fig4.PNG" description="" class="full-image" %}

비디오를 만들어와서 보여주는 사람도 있고 짧고 빠르게 ppt를 발표하는 사람도 있고 다양한 방식으로 본인의 연구를 적극적으로 홍보를 하는데 굉장히 인상 깊었습니다. 
본 프로그램은 실시간으로 유튜브를 통해 스트리밍이 진행되었으며 해당 영상은 
<a href="https://www.youtube.com/watch?v=CV_14aUBxsI" target="_blank"><b> 해당 링크 </b></a>
에서 확인이 가능합니다. 
이 프로그램을 통해 앞으로 펼쳐질 많은 논문 발표 중에 어떤 발표가 재미가 있을 지, 어떤 논문을 숙소에서 미리 읽어볼 지 등에 대한 감을 잡을 수 있어서 개인적으로 굉장히 인상 깊었습니다. 
실제로도 이 프로그램에서 인상깊게 들은 논문을 호텔에서 읽어보고, 발표장에서도 듣기도 하였습니다. 
이 논문은 다음 포스팅에서 자세히 다루도록 하겠습니다.

### - NVIDIA Turing Architecture 공개
SIGGRAPH 2018의 화두는 단언컨대 NVIDIA turing 아키텍처의 공개라고 할 수 있을 것 같습니다. 
NVIDIA의 젠슨황 CEO가 SIGGRAPH 2018에서 최초로 turing 아키텍처 기반의 GPU를 공개하였고, 굉장히 주목을 받게 되었습니다. 
사실 하드웨어에 관심이 없는 분들은 이게 뭔데? 하실 수 있는데 저처럼 하드웨어에 관심이 많은 사람들에겐 중요한 자리였습니다. 

사실 NVIDIA GPU에 대해 설명을 하자면 정말 길게 설명이 필요해서 단순하게 설명을 드리면 마치 인텔의 CPU에 세대가 존재하듯이 NVIDIA의 GPU에도 세대가 존재합니다. 
예를 들어 인텔 코어 i 시리즈를 예로 들면, 4세대(하스웰), 5세대(브로드웰), 6세대(스카이레이크), 7세대(카비레이크), 8세대(커피레이크) 등등 세대가 올라갈수록 미세 공정 도입, base clock의 전반적인 성능 향상 등 성능이 좋아진다고 알려져 있습니다. 
마찬가지로 NVIDIA GPU도 세대가 올라갈수록 전반적인 성능이 향상된다고 할 수 있습니다. 
최근 세대별 대표적인 GPU를 정리하면 다음과 같습니다.

-	9세대(Kepler)
     - GeForce 600 series (ex, GT 640, GTX 650 Ti, etc)
     - GeForce 700 series (ex, GT 740, GTX 750 Ti, etc)
     - GeForce TITAN (ex, GTX TITAN, TEX TITAN Z, etc)
-	10세대(Maxwell)
     - GeForce 900 series (ex, GTX 980, GTX 980 Ti, etc)
     - GeForce TITAN X (ex, GTX TITAN X)
-	11세대(Pascal)
     - GeForce 10 series (ex, GTX 1060, GTX 1080 Ti, etc)
     - TITAN X/Xp (ex, TITAN X/Xp)
-	12세대(Volta)
     - TITAN V (ex, TITAN V)
-	13세대(Turing)
     - GeForce 20 series (ex, RTX 2070, RTX 2080 Ti, etc)
     - TITAN RTX (ex, TITAN RTX)

저희가 알 만한, 저희 주변에서 흔히 볼 수 있는 GPU들은 대부분 9세대 ~ 12세대에 속해 있으며 13세대인 **Turing** 이 바로 이번 SIGGRAPH 2018에서 최초로 공개된 아키텍처라 할 수 있습니다. 

가장 도드라지는 특징은 Ray-Tracing을 위한 **RT Core** 가 공개가 되었고, 최초로 GPU로 실시간으로 Ray-Tracing을 할 수 있음을 보였습니다. 
또한 딥러닝 연산과 관련이 있는 **Tensor Core** 도 이전 버전보다 더 발전된 형태로 공개가 되었습니다. 
제가 데모에서 봤던 GPU를 찾아보니 GPU 메모리가 무려 48GB로 NVLink를 이용하면 무려 96GB라는 어마어마한 메모리 사양을 보이고, 초당 10기가 ray의 ray-Tracing 성능을 보이기도 하며, 저희가 관심있어하는 Tensor Core도 576개가 부착되어 있는 굉장히 하이엔드 GPU인 **쿼드로 TX 8000** 이었습니다. 
스펙에 놀랐고 성능에 놀랐고 가격에 놀랐습니다. (무려 1만 달러!) 

어쩌다 보니 NVIDIA의 마케팅 직원과 같이 글을 쓰고 있는데, 저에게 돌아오는 건 아무것도 없다는 사실을 말씀 드리며 NVIDIA 관련 내용은 여기까지 하도록 하겠습니다. 
SIGGRAPH 2018에서 발표된 영상은 
<a href="https://www.youtube.com/watch?v=jY28N0kv7Pk" target="_blank"><b> 해당 링크 </b></a>
에서 확인하실 수 있습니다.

### - 그 외..
이 외에도 다양한 프로그램들이 진행이 되었습니다. 
같은 시간대에 두 건물에서 동시에 여러 프로그램이 진행이 되다 보니 못 가본 프로그램들이 많습니다. 
대표적으로 Art 관련 프로그램들이 못 가본 프로그램들이고, 이 외에도 여러 기업을 초청해서 진행하는 Job Fair, 여러 제품들을 전시하고 설명하는 production gallery, 논문으로 발표한 기술들을 실시간으로 데모로 보여주는 Real-time live 등 다양한 프로그램들이 있었습니다. 

전체 프로그램 일정은 
<a href="https://s2018.siggraph.org/wp-content/uploads/2018/06/s2018_advance_program.pdf" target="_blank"><b> 해당 링크 </b></a>
에서 확인하실 수 있습니다. 
만약 추후에 SIGGRAPH에 참석을 하실 계획이 있으신 분들은 일정이 정해지면 스케쥴링을 잘 하셔서 본인이 좋아하는 프로그램 위주로 들으시는 것을 추천 드립니다. 

## 결론
이번 포스팅에서는 SIGGRAPH라는 학회에 대해 간단하게 소개를 드리고, 제가 보고 느낀 점들을 정리해보았습니다. 
다음 포스팅에서는 제가 학회장에서 인상깊게 들었던 논문 여러 편을 간략하게 소개 드리고, 그 중 한 편(Dataset and metrics for predicting local visible differences)은 심도 있게 리뷰를 진행하도록 하겠습니다. 
읽어주셔서 감사합니다!

[(다음 포스팅 보기)]({{ page.url }}/../siggraph2018-review-2.html)

