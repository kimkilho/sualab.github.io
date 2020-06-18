---
layout: post
title: "파이토치에도 보일러플레이트가 스치운다"
date: 2020-06-18 00:00:00 +0900
author: cho_dongheon
categories: [Development]
tags: [pytorch-lightning, boilerplate]
comments: false
name: pytorch-boilerplate
image: exp.png
---

## TL;DR

{% include image.html name=page.name file="tldr.png" description="" class="full-image" %}

"OmegaConf"로 실험 설정을 관리하고 "Pytorch-Lightning"으로 실험 코드를 구성하고, "Microsoft NNI"+"Tensorboard"로 실험을 기록하는 과정을 "Docker" 환경을 구축해서 하자!

## 딥러닝 "실험"이란?

연구는 관찰, 가설 설정, 실험 그리고 반복의 과정입니다. 관찰과 가설 설정은 연구자의 번뜩이는 아이디어를 갈고 닦으며 사고 과정에서 이루어질 수 있으나 결국 '실험'을 통해 여러 동료 연구자 및 통계 모델들에게 검증을 받아야 비로소 논문이 나옵니다. (혹은 제품화까지 갈 수 있겠네요!)

그러면 딥러닝을 위한 실험은 어떻게 구성되어 있을까요? N명의 연구자가 있으면 N개의 연구 방법론들이 있겠습니다만, 필자는 딥러닝 실험을 "실험 도구", "실험 수행" 그리고 "실험 환경"으로 나누어 접근하였습니다.

{% include image.html name=page.name file="exp.png" description="실험실의 삼요소" class="full-image" %}

"실험 도구"는 우리의 가설을 현실에 구현할 매체입니다. 딥러닝에서는 실험을 수행할 Pytorch, Tensorflow, MXNet 등으로 생각해볼 수 있겠네요. 

"실험 수행"은 독립, 통제 변인과 그에 따른 종속 변인의 변화를 기록하며 가설을 수정하는 과정입니다. 코드에서 사용되는 여러 설정들을 관리하고 Loss/Metric들을 최적화시키는 프로세스라고 여겨집니다.

마지막으로 "실험 환경"은 실험이 진행되는 환경입니다. 실험자는 패키지를 설치 후 고정된 환경에서 실험합니다만, 오픈 소스 기여자분들 덕분에 패키지는 지속해서 업데이트되어 인터페이스 혹은 작동 방법 등이 변경되곤 합니다.

### [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) : 실험 도구

{% include image.html name=page.name file="pl.png" description="Pytorch 코드를 Pytorch Lightning으로 변환하는 도식" class="full-image" %}

"Pytorch Lightning"은 기존의 Pytorch 코드를 Research/Engineering/Non-essential 3가지로 구분하여 모델 정의 및 학습에 관련된 Research 코드 작성 외의 GPU 설정, 로깅, 실험 설정 등은 기본적으로 제공하여 적은 수정으로 사용할 수 있도록 제공합니다.

> Research Code == 'LightningModule'

```python
import pytorch-lightning as pl

class PL(pl.LightningModule):
    def __init__(self, network: dict, dataloader: dict):
        super().__init__()
        self.network = network['network'] # nn.Module
        self.hparams = dict(network['network_option']) # Dict Configs
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, batch, network): # 네트워크 forward 정의
        img = batch[0]
        Y = batch[1]
        pred = network(img)
        return pred, self.loss(pred.float(), Y.long())
    
    def training_step(self, batch, batch_nb): # 1 train iteration 정의
        pred, loss = self.forward(batch, self.network)
        return {'loss' : loss, 'progress_bar' : {'train_loss' : loss }}
    
    def validation_step(self, batch, batch_nb): # 1 val iteration 정의
        pred, loss = self.forward(batch, self.network)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs): # 1 val epoch 정의
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def train_dataloader(self): # train dataloader 정의
        return self.dataloader["train"]

    def configure_optimizers(self): # optimizer 정의
        return torch.optim.Adam(self.parameters(), lr=0.02)
		...
```

Research 부분인 `LightningModule` 은 기존 pytorch의 `nn.Module` 에 데이터, 로스, 옵티마이저 설정을 추가한 모듈입니다.  `__init__` `forward` 에 네트워크 구조를 정의하고, 추가적으로 `training_step` `train_dataloader` `configure_optimizers`  등의 함수들을 오버 라이딩하여 학습 데이터 및 옵티마이저를 정의하여 줍니다.  `training_step` , `validation_step`, `validataion_epoch_end`에서 "log" key로 리턴하는 내용은 Logger에 batch step 기준으로 로깅되고, "progress_bar" key로 리턴하는 내용은 terminal에서 progress bar 프로그램인 [tqdm](https://github.com/tqdm/tqdm)을 통해 출력됩니다.

{% include image.html name=page.name file="tqdm.png" description="tqdm을 통해 출력되는 예시" class="full-image" %}

LightningMoudule에 관해 더 자세한 내용은 [라이트닝 모듈 링크](https://pytorch-lightning.readthedocs.io/en/0.7.6/lightning-module.html)를 참고해주세요

> Engineering & Non-essential 

```python
from pytorch-lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger

...
trainer = Trainer(
    logger=TensorBoardLogger(save_dir="./Logs", name="exp"), # 실험 로거 정의
    gpus=1, # 사용할 gpu 개수 
    max_epochs=100, # 최대 epoch
    log_save_interval=1, # log 저장 간격
)
...
trainer.fit(pl) # train, validation 실행
trainer.test(pl) # test 실행

```

Engineering 관련 내용을 처리하는 `Trainer`는 다양한 option들이 있습니다만, 기본적으로 logger와 gpu, epoch을 설정하여 기존의 코드에서 사용되었던 gpu 설정 및 for 반복문, metric 로깅 과정을 일부 생략할 수 있습니다. 

이후 `trainer.fit`을 통해 모델의 train/val 데이터를 이용한 학습 및 평가가 진행되고, `trainer.test`를 실행하여 test 데이터에 대한 최종 평가를 수행할 수 있습니다. 이 과정은 학습 중 tqdm을 통해 진행도가 출력되고 logger에 추가로 기록을 합니다. 위의 코드에서는 TensorBoardLogger를 사용하였으나 [Comet](https://www.comet.ml/), [Neptune](https://neptune.ai/), [WanDB](https://www.wandb.com/) 같은 다양한 로거들을 지원하고 있습니다

gpu 분산 학습인 distributed training, TPU 사용 옵션, Nvidia에서 개발한 뉴럴넷 최적화 툴 [APEX](https://github.com/NVIDIA/apex)에 관련된 설정 등 여러 유용한 옵션들이 있으니 [https://pytorch-lightning.readthedocs.io/en/0.7.6/trainer.html](https://pytorch-lightning.readthedocs.io/en/0.7.6/trainer.html#amp-level)를 참고하시길 바랍니다.

* 앗! 잠깐만!

    `pytorch_lightning.seed_everything(seed)` 를 코드 처음에 넣어두면 seed를 고정해 실험 reproducible를 확보할 수 있다는 사실~

### [MS NNI](https://github.com/microsoft/nni), [OmegaConf](https://github.com/omry/omegaconf) : 실험 수행

위의 Pytorch Lightning 코드만 모두 작성하더라도, 실험 준비는 완료된 것이나 다름없습니다! 하지만, 실험은 한번 수행하고 끝나는 것이 아닌 무한 번의 실험을 통해 도출된 결과들을 분석하여 최적의 값을 찾는 것이 우리의 일입니다. 실험 수행 중 조금이나마 실험 관리 자동화를 도와줄 도구 Microsoft Neural Network Intelligence와 OmegaConf를 소개합니다.

> Microsoft Neural Network (aka NNI)

NNI는 머신 러닝 과정 중 Hyper-parameter Tuning, feature engineering, neural architecture search 등의 과정을 도와주는 AutoML 오픈 소스 프로젝트입니다. 다양한 기능이 포함된 멋진 툴킷이지만, 오늘은 Hyper-parameter Tuning 정도만 소개를 드리려고 합니다.

NNI는 3가지 과정을 통해 사용할 수 있습니다.

{% include image.html name=page.name file="nni.png" description="MS NNI를 코드에 적용하는 3가지 단계" class="full-image" %}

첫 번째 과정으로는 Tuning을 적용할 Search Space 정의입니다. 작게는 learning rate, batch size부터 크게는 네트워크 종류 선택까지 적용해볼 수 있겠네요. Auto Searcher (Grid, GP, PPO 등)의 종류에 따라 다르지만, search 범위는 범주형 `choice`,  `uniform` , `randint` 정도의 타입으로 나눌 수 있겠습니다. 해당 설정은 `.json`타입으로 작성하시면 됩니다.

```python
{
    "train.batch_size":{"_type":"choice","_value":[32,64,128]},
    "network.version":{"_type":"choice","_value":["1_0", "1_1"]}
}
```

그다음에 search parameter를 코드에 연결하고 최적화할 목표 metric을 설정해야 합니다. 필수적으로는 `report_final_result`와 `get_next_parameter`를 사용하여야 하고, 선택적으로 학습 과정 중 관찰하고 싶은 metric을 `report_intermediate_result`로 NNI 플랫폼에 기록해볼 수 있습니다.

```python
def _main(cfg=dc.DefaultConfig) -> None:
    params = nni.get_next_parameter() # 정의된 search space에서 next step의 config 호출
    params = search_params_intp(params)
    cfg = OmegaConf.structured(cfg)
    args = OmegaConf.merge(cfg, params) # nni search config와 기본 config 병합

    ml = main_pl.MainPL(
        args.train, args.val, args.test, args.hw, args.network, args.data, args.opt, args.log, args.seed
    )
    final_result = ml.run()
    nni.report_final_result(final_result) # nni에서 추적할 log 기록
```

마지막으로는 AutoML을 할 때 사용할 Tuner와 탐색 시간 및 자원 설정입니다. `.yml` 타입으로 작성하면 되고, 필수적으로는 `useAnnotation` 을 false로 지정하고 `searchSpacePath` 의 값으로 탐색할 값을 불러옵니다. 추가적인 옵션으로 `trialConcurrency` 로 동시에 수행할 실험, `maxExecDuration` `maxTrialNum` 으로 탐색 시간 및 횟수를 설정할 수 있습니다. 그리고 제일 중요한 `tuner:builtinTuerName` 을 통해 파라미터 탐색 시 사용할 Tuner를 고를 수 있습니다. Tuner는 기본적으로 모든 조합을 찾는 Grid Search, 강화학습 기반의 PPO Tuner, Random Search 등의 다양한 옵션들이 있습니다. 더욱 다양한 튜너에 대한 옵션과 사용법들은 [https://nni.readthedocs.io/en/latest/hyperparameter_tune.html](https://nni.readthedocs.io/en/latest/hyperparameter_tune.html)에서 확인해주시면 됩니다!

```python
authorName: davinnovation
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10

trainingServicePlatform: local #choice: local, remote, pai
searchSpacePath: config/search.json

useAnnotation: false
tuner:
  builtinTunerName: GridSearch
trial:
  command: python run_nni.py
  codeDir: .
  gpuNum: 1
```

위의 단계를 마치고 NNI을 실행시킨다면 다음과 같은 결과를 `localhost:8080` 에 접속하시면 현재 탐색 실행 상황 및 최적 값 등을 확인하실 수 있습니다.

{% include image.html name=page.name file="nni2.png" description="NNI를 실행 후 parameter를 grid search하고 loss 결과에 매핑한 그래프" class="full-image" %}

> OmegaConf

딥러닝 프로젝트를 진행하다 보면 초반에는 batch size, epoch 등의 적은 옵션들만 설정하여 하드 코딩으로 관리하기 괜찮지만, 설정의 종류가 시간이 지나며 늘어나거나 cli, config.json 등등 옵션들에 대한 entry point가 많아지면 관리하기도 힘든데요, 이를 `OmegaConf` 를 통해 관리하면 조금 더 편하게 사용할 수 있습니다. 

여러 사용법이 있습니다만, 필자는 데이터를 관리하는데 기본적인 매니징 기능이 추가된 [dataclass](https://docs.python.org/ko/3/library/dataclasses.html) 를 기반으로 설정 관리하는 것을 선호합니다. 우선 config.py에 관리하고자 하는 설정을 입력합니다.

```python
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class TrainConfig:
    batch_size: int = 256
    epoch: int = 5
...
@dataclass
class OptConfig:  # flexible
    opt: str = "Adam"
    lr: float = 1e-3

@dataclass
class DefaultConfig:
    train: TrainConfig = TrainConfig()
    val: ValConfig = ValConfig()
    test: TestConfig = TestConfig()
    hw: HWConfig = HWConfig()
    network: NetworkConfig = NetworkConfig()
    data: DataConfig = DataConfig()
    opt: OptConfig = OptConfig()
    log: LogConfig = LogConfig()
    seed: str = 42
```

그리고 위의 Config.py를 main에서 호출하여 기본 config로 사용하고, entry point를 다양화할 수 있도록 `merge_with_cli` 등을 통해 수정할 수 있도록 합니다.

```python
...
def _main(cfg=dc.DefaultConfig) -> None:
    args = OmegaConf.structured(cfg)
    args.merge_with_cli()
...
```

OmegaConf의 장점은 계층적인 설정을 다양한 entry point에서 (dictionary, dataclass, cli args 변환, list etc) 입력받을 수 있고 통합할 수 있도록 지원합니다. 사용법을 추가로 확인하고 싶으시다면 [omegaconf 2.0 slide](https://docs.google.com/presentation/d/e/2PACX-1vT_UIV7hCnquIbLUm4NnkUpXvPEh33IKiUEvPRF850WKA8opOlZOszjKdZ3tPmf8u7hGNP6HpqS-NT5/pub?start=false&loop=false&delayms=3000&slide=id.g84632f636b_10_519)를 확인해주세요

* 앗! 잠깐만!

    아쉽게도 OmegaConf는 argparse나 python-fire 등의 argument manager들이 기본적으로 지원하는 auto doc을 생성하지 않아 —help 명령어를 사용하는 것이 어렵습니다 ㅠㅠ

### [Anaconda](https://www.anaconda.com/products/individual), [Docker](https://www.docker.com/) : 실험 환경

딥러닝 실험 코드를 작성하고, 설정 관리 및 자동 탐색 툴까지 붙였는데 더 해야 하는 것이 있을까요? 이번 툴은 필수는 아니나 추후에 다시 코드 실행 환경을 복구하기 위한 Docker와 Anaconda를 통한 파이썬 라이브러리 관리를 소개드립니다.

{% include image.html name=page.name file="meme.png" description="논문 코드가 바로 실행될 때.jpg" class="full-image" %}

논문을 탐색하며 github 링크가 있을 때는 정말 기쁩니다만, repo를 pull 해서 실행 시 requirements가 없으면 당황스럽죠.  많은 연구원들이 실험할 때 여러 패키지를 사용하며 코드를 작성합니다만, 사용되었던 패키지들은 업데이트가 되며 몇 년 뒤에는 예전 코드를 실행할 수 없을 때가 종종 생기게 됩니다. 이럴 때를 위해 실험 환경을 정확하게 기술해두면 좋지만, 연구자로서는 local의 환경을 계속 확인하며 공유하는 게 번거로운 일이 됩니다.

> Anaconda

[Anaconda](https://www.anaconda.com/products/individual)는 데이터 과학을 위한 Python 패키지들과 라이브러리들을 모아두고 환경 관리 및 설치를 도와주는 소프트웨어입니다. 

Anaconda를 설치하시고 path를 설정해주시면 prompt 창에서 가장 왼쪽에 `(base)`로 표시가 됩니다. 이는 현재 base라는 기본 환경을 가지는 python을 실행하기 위해 준비되었다는 의미입니다.

{% include image.html name=page.name file="conda.png" description="conda가 준비되었을 때의 콘솔창" class="full-image" %}

한 개의 환경에 모든 라이브러리를 설치하지 않고 각 프로젝트를 위한 환경을 만들어서 관리하기 위해 `conda create -n torch_py37 python=3.7 anaconda` 명령어로 torch_py36이라는 새로운 환경을 만들어줍니다. 그 다음 `conda activate torch_py37` 를 통해 환경을 교체할 수 있습니다.

그리고 `conda install`을 통해 여러 패키지를 설치하시고, `conda env export > env.yaml` 로 환경을 공유할 수 있습니다. 그리고 다른 anaconda에서는 `conda create env -f env.yaml` 를 통해 재사용할 수 있게됩니다!

> Docker

*"도커 컨테이너"는 일종의 소프트웨어를 소프트웨어의 실행에 필요한 모든 것을 포함하는 완전한 파일 시스템 안에 감싼다. 여기에는 코드, 런타임, 시스템 도구, 시스템 라이브러리 등 서버에 설치되는 무엇이든 아우른다. 이는 실행 중인 환경에 관계없이 언제나 동일하게 실행될 것을 보증한다. - 위키피디아*

Docker는 소프트웨어를 실행하기 위한 환경 관리 툴입니다. 

필자는 floydhub의 도커 이미지를 기반으로 실험을 진행합니다. 각 딥러닝 프레임워크를 위한 이미지들이 준비되어 있고 추가 패키지들과 CUDA 버전 별 관리가 되어있어 많은 추가 패키지들을 설치하지 않고 기본으로 사용하기 좋습니다 - [https://hub.docker.com/u/floydhub](https://hub.docker.com/u/floydhub)

위의 Docker Image를 확장하여 자신만의 Dockerfile을 만들고 관리하는 것도 쉽습니다. 

```docker
FROM floydhub/pytorch:1.5.0-gpu.cuda10cudnn7-py3.55

RUN mkdir app
WORKDIR app
RUN git clone https://github.com/davinnovation/pytorch-boilerplate
WORKDIR pytorch-boilerplate

RUN pip install -r requirements.txt
```

Dockerfile을 만들고 base로 사용할 image를 FROM으로 기술합니다. 그리고 그 뒤에는 RUN을 통해 추가로 필요한 파이썬 패키지, 소프트웨어 등을 설치하는 명령어를 적어두면 어느 컴퓨터나 동일한 실험 환경을 사용할 수 있습니다!

Docker에 관한 자세한 내용 등은 [https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html) 여기를 따라가 보시면 좋겠습니다!

* 앗! 잠깐만!

    그리고 아직은 윈도우 Docker에서는 GPU 사용이 불가능하여 리눅스를 사용하는 것을 권장해 드립니다만, WSL 2라는 윈도우 10의 가상화 머신이 곧 GPU를 지원한다고 하니 Follow UP!

### Example Code

지금까지 살펴본 코드 과정의 전체 플로우를 지닌 프로젝트 예제를 보고 싶으시다면 [https://github.com/davinnovation/pytorch-boilerplate/](https://github.com/davinnovation/pytorch-boilerplate/)을 참고해주세요!

[![image](https://user-images.githubusercontent.com/3917185/84723043-ac25e380-afbf-11ea-9116-fbabd47b5cc0.png)](https://github.com/davinnovation/pytorch-boilerplate/)

## 참고. 보면 좋은 것들

- AI 연구자를 위한 클린코드 작성법

    [https://www.slideshare.net/KennethCeyer/ai-gdg-devfest-seoul-2019-187630418](https://www.slideshare.net/KennethCeyer/ai-gdg-devfest-seoul-2019-187630418)

- 추가하면 좋을 것들

    [OpenPAI](https://openpai.readthedocs.io/), [Horovord](https://github.com/horovod/horovod): 효과적인 학습을 위한 Distributed Training 플랫폼입니다

    [black](https://github.com/psf/black) : 코드 포맷팅

    [line_profiler](https://github.com/psf/black), [profiling](https://github.com/what-studio/profiling) : 코드 성능 프로파일링

- 위 프로젝트들의 대체

    Pytorch lightning → [Pytorch Ignite](https://github.com/pytorch/ignite) : Pytorch 그룹의 공식 High Level Framework입니다

    NNI → [Ray.Tune](https://docs.ray.io/en/master/tune.html) : 널리 사용되는 Parameter 튜너입니다

    OmegaConf → [python-fire](https://github.com/google/python-fire) : google에서 만든 옵션에 대한 명시 없이 class/function을 기반으로 cli interface를 만들어 주는 라이브러리입니다

## 다른 글도 읽으러 가기

[DCGAN을 이용한 이미지 생성](https://research.sualab.com/introduction/practice/2019/05/08/generative-adversarial-network.html)

[머신러닝 모델에 대한 해석력 확보를 위한 방법](https://research.sualab.com/introduction/2019/08/30/interpretable-machine-learning-overview-1.html)