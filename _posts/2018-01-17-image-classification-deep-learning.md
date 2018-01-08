---
layout: post
title: "이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기"
date: 2018-01-17 09:00:00 +0900
author: kilho_kim
categories: [machine-learning, computer-vision]
tags: [classification, alexnet]
comments: true
name: image-classification-deep-learning
---

지금까지 딥러닝과 이미지 인식 문제에 대해서 알아보았습니다. 해결하고자 하는 문제(이미지 인식)의 개괄을 살펴보았고 문제 해결을 위한 도구(딥러닝)에 대해 알아보았으니, 이제는 좀 더 구체적으로 이미지 인식 문제에 딥러닝을 직접 적용한 사례를 하나 제시하고, 이를 실제 구현 코드와 함께 소개해 드리고자 합니다. 지금까지의 글들이 대부분 '개념적인' 이야기들 위주였다면, 본 글에서는 코드에 기반한 '실제적인' 내용이 다뤄진다고 이해하시면 될 것 같습니다.

- **주의: 본 글은 아래와 같은 분들을 대상으로 합니다.**
  - 딥러닝 알고리즘의 기본 구동 원리 및 정규화(regularization) 등의 테크닉에 대한 기초적인 내용들을 이해하고 계신 분들
  - Python 언어 및 TensorFlow의 기본적인 사용법을 알고 계신 분들
- 본 글에서는, 딥러닝 모델 및 알고리즘 구현을 위한 하나의 방식을 제시합니다. 이는 새로운 딥러닝 테크닉이 등장하였을 때, 여러분들이 사용하던 기존 모델 혹은 알고리즘에 빠르고 효과적으로 적용할 수 있도록 하기 위함이며, 그와 동시에 딥러닝 모델과 알고리즘의 작동 방식을 더 잘 이해할 수 있도록 하기 위함입니다.
- 본 글에서 구현한 AlexNet은, 원본 AlexNet 논문의 셋팅과 일부 다른 부분이 존재합니다. 이러한 부분을 본문 중간중간에 명시하였습니다.
- 본문에서는 전체 구현체 중 핵심적인 부분을 중심으로 설명합니다. 전체 구현체 코드는 수아랩의 GitHub 저장소(*TODO: 링크 추가*)에서 자유롭게 확인하실 수 있습니다. 


## 서론

<a href="{{ site.url }}/computer-vision/2017/11/29/image-recognition-overview-1.html" target="_blank">\<이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로\></a>에서 언급한 바에 따르면, PASCAL VOC Challenge의 규정에 의거하면 이미지 인식 분야에서 총 3가지 문제를 다룬다고 하였습니다. 이번 글에서는 이들 문제 중 가장 단순한 축에 속하는 **Classification** 문제의 한 가지 사례를 가져오고, 이를 딥러닝 기술로 해결하는 과정을 여러분께 안내하고자 합니다.

본격적인 글의 전개에 앞서 중대한 사항을 하나 말씀드려야 하는데, 본 글이 *(1) 딥러닝에서의 심층 신경망 모델을 학습시키는 러닝 알고리즘에 대한 기초적인 이해가 있으며,* *(2) Python 언어 및 TensorFlow의 기초를 알고 있는* 분들을 타겟으로 한다는 점입니다. 만약 이게 갖춰지지 않은 분들이 계시다면, 아쉽지만 온/오프라인 상에 좋은 교육 자료들이 많이 있으니 이들을 먼저 공부하고 오시길 권해 드립니다. 

먼저 본 글에서 다룰 Classification 문제로는, 상대적으로 단순하면서도 많은 분들의 흥미를 끌 만한 주제인 '*개vs고양이 분류*' 문제를 다룰 것입니다. PASCAL VOC Challenge의 Classification 문제의 경우 주어진 이미지를 총 20가지 클래스(class) 중 하나(혹은 복수 개)로 분류하는 것이 목표였다면, '개vs고양이 분류' 문제에서는 주어진 이미지를 '개'와 '고양이'의 두 가지 클래스 중 하나로 분류하는 것이 목표라는 점에서 훨씬 단순하다고 할 수 있으며, 귀여운 개들과 고양이들의 이미지를 보는 재미가 쏠쏠하다(?)고 할 수 있습니다.

그리고 이 문제를 해결하기 위해, 2012년도 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회에서 각광을 받은 대표적인 컨볼루션 신경망(convolutional neural network)인 **AlexNet** 모델을 채택하여 학습을 진행할 것입니다. 오늘날 AlexNet보다 더 우수한 성능을 발휘한다고 알려져 있는 딥러닝 모델들이 많이 나와 있음에도 AlexNet을 쓰는 이유는, AlexNet만큼 검증이 많이 이루어진 딥러닝 모델이 드물고, 다양한 이미지 인식 문제에서 AlexNet만을 사용하고도 준수한 성능을 이끌어냈다는 사례들이 많이 보고되어 왔기 때문입니다.

### 굳이 왜?

이 쯤에서, 딥러닝을 어느 정도 알고 계시는 분들이라면 틀림없이 아래와 같은 의문을 제기하실 것 같습니다. 

> 온라인 상에 수많은 AlexNet 구현체들이 존재하는데, 굳이 이걸 여기에서 직접 제작해야 할까?

어느 정도는 일리가 있는 의문입니다. 그런데, 딥러닝 모델 구현체들을 검색하려고 조금이라고 시도해보신 분들은 아시겠지만, 온라인 상에 존재하는 구현체들을 자세히 살펴보면 하나의 파일 안에 데이터셋, 모델, 알고리즘 등과 관련된 부분들이 서로 얽히고설켜 커다란 한 덩어리로 뭉쳐있는 경우가 상당히 많습니다. 이렇게 하는 것이 맨 처음에 구현체를 신속하게 완성하고 학습 결과를 빨리 관찰하는 데 있어서는 유용한 것이 사실이나, 여기에는 몇 가지 치명적인 단점이 존재합니다.
  
우선, 이는 초심자 입장에서 딥러닝에 대해 이해하는 데 있어 효과적이지 못합니다. 보통 딥러닝의 기초 학습을 갓 마치신 분들의 머릿속에는 이런저런 개념들이 충분히 체계화되지 않은 채로 존재할 것입니다. 이 상황에서 '이제 코드 좀 짜 볼까?' 하고 TensorFlow 등의 딥러닝 프레임워크로 짜여진 예시 구현체를 찾아볼 것인데, 한 덩어리로 된 구현체들만을 계속 보다 보면 머릿속의 혼잡한 개념들이 여전히 정리가 되지 않은 형태로 남아있게 될 가능성이 높습니다.

또한, 이는 구현체 코드에 대한 유지/보수의 측면에서도 좋지 못합니다. 반복 실험 과정에서의 모델의 성능 향상을 위해, 여러분들은 기존에 마련해 놓은 구현체에 최근에 핫하다고 나온 이런저런 테크닉들을 하나둘씩 추가로 적용하여 커스터마이징(customizing)을 할 것인데, 그러다 보면 기존 코드가 걸레짝(?)처럼 되는 경우가 비일비재합니다. 그러다가 어딘지 모를 지점에서 문제가 생겨 학습이 잘 이루어지지 않는 상황이 찾아오면 문제는 더 심각해지는데, 이 얽히고설킨 코드들을 샅샅이 훑어보는 과정에서 여러분들은 극도의 스트레스에 시달리게 될 가능성이 높습니다.

지난 <a href="{{ site.url }}/machine-learning/2017/10/10/what-is-deep-learning-1.html" target="_blank">\<딥러닝이란 무엇인가?\></a>에서 *딥러닝은 머신러닝의 세부 방법론에 불과하다*고 하였으며, <a href="{{ site.url }}/machine-learning/2017/09/04/what-is-machine-learning.html" target="_blank">\<머신러닝이란 무엇인가?\></a>에서는 **머신러닝의 핵심 요소**로 *데이터셋(data set)*, *러닝 모델(learning model)*, *러닝 알고리즘(learning algorithm)* 등이 있다고 하였습니다. 여기에 학습된 러닝 모델에 대한 *성능 평가(performance evaluation)*를 추가하면, 아래와 같은 리스트가 완성됩니다.

#### 이미지 인식 문제를 위한 딥러닝의 기본 요소

- 데이터셋
- 성능 평가
- (딥)러닝 모델
- (딥)러닝 알고리즘

머신러닝의 핵심 요소를 고려하여, 딥러닝 구현체를 위와 같이 총 4가지 기본 요소로 구분지어 이해하고자 시도한다면, 딥러닝 관련 개념들을 이해하고 이를 기반으로 자신만의 구현체를 만드는 데 매우 유용합니다. 뿐만 아니라, 새로운 딥러닝 관련 테크닉이 나오게 되더라도 그것이 위 4가지 요소 중 어느 부분에 해당하는 것인지를 먼저 파악하게 되면, 그것을 구현하는 시간을 그만큼 단축시킬 수 있습니다. 가령 'DenseNet은 기존 모델에 skip connections를 무수히 많이 추가한 것이므로, 기존 러닝 모델에 이를 추가하면 되겠구나!' 내지는 'Adam은 기존의 SGD에 adaptive moment estimation을 추가한 것이므로, 기존 러닝 알고리즘에 이를 추가하면 되겠구나!' 하는 식으로 생각할 수 있을 것입니다.

그러면 지금부터, 위에서 언급한 딥러닝의 4가지 기본 요소를 기준으로 삼아, '개vs고양이 분류' 문제 해결을 위해 직접 제작한 AlexNet 구현체를 소개해 드리도록 하겠습니다.


## (1) 데이터셋: Asirra Dogs vs. Cats dataset

개vs고양이 분류 문제를 위해 사용한 데이터셋의 원본은 **The Asirra dataset**이며, 본 글에서 실제 사용한 데이터셋은 데이터 사이언스 관련 유명 웹사이트인 Kaggle에서 제공하는 competitions 항목 중 <a href="https://www.kaggle.com/c/dogs-vs-cats" target="_blank">Dogs vs. Cats</a>로부터 가져온 것입니다.

{% include image.html name=page.name file="dogs-cats-examples.png" description="Asirra Dogs vs. Cats 데이터셋 내 개/고양이 이미지 예시" class="full-image" %}

원본 데이터셋은 학습 데이터셋(training set) 25,000장, 테스트 데이터셋(test set) 12,500장으로 구성되어 있으나, 이 중 학습 데이터셋에 대해서만 레이블링(labeling)된 채로 제공되고 있습니다. 본 글에서의 개vs고양이 분류 문제 셋팅을 위해, 원본 학습 데이터셋 중 랜덤하게 절반 크기만큼 샘플링(sampling)하여 이 부분(12,500장)을 학습 데이터셋으로, 나머지 절반에 해당하는 부분(12,500장)을 테스트 데이터셋으로 재정의하였습니다. 

이미지 크기는 가로 42~1050px, 세로 32~768px 사이에서 가변적입니다. 개vs고양이 분류 문제용 데이터셋이므로, 자연히 클래스는 0(고양이)과 1(개)의 이진(binary) 클래스로 구성되어 있습니다.

*TODO: 데이터셋 다운로드 링크 추가*


### datasets.asirra 모듈

`datasets.asirra` 모듈은, 데이터셋 요소에 해당하는 모든 함수들과 클래스를 담고 있습니다. 이들 중에서, 디스크로부터 데이터셋을 메모리에 로드하고, 학습 및 예측 과정에서 이들을 미니배치(minibatch) 단위로 추출하는 부분을 중심으로 살펴보도록 하겠습니다.

#### read_asirra_subset 함수

```python
def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    """
    Load the Asirra Dogs vs. Cats data subset from disk
    and perform preprocessing for training AlexNet.
    :param subset_dir: str, path to the directory to read.
    :param one_hot: bool, whether to return one-hot encoded labels.
    :param sample_size: int, sample size specified when we are not using the entire set.
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """
    # Read trainval data
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    if sample_size is not None and sample_size < set_size:
        # Randomly sample subset of data when sample_size is specified
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
        set_size = sample_size
    else:
        # Just shuffle the filename list
        np.random.shuffle(filename_list)

    # Pre-allocate data arrays
    X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
    y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # Convert labels to one-hot vectors, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set
```

`read_asirra_subset` 함수에서는 원본 Asirra 데이터셋을 디스크로부터 읽어들인 뒤, 각 이미지를 $$256\times256$$ 크기로 일괄적으로 리사이징(resizing)하고, 이들을 numpy의 `np.ndarray` 형태로 반환합니다. 또한, 레이블의 경우 `one_hot` 인자가 `True`인 경우 one-hot encoding 형태로 변환하여 반환하도록 하였습니다.

이 때 주의할 점은, 위 함수를 사용하여 전체 학습 데이터셋을 메모리에 로드하고자 할 경우, *적어도 16GB 이상의 메모리를 필요로 한다*는 점입니다. 메모리 사양이 이에 못 미치는 분들을 위한 궁여지책(?)으로 함수에 `sample_size` 인자를 추가하였는데, 이를 명시할 경우 원본 데이터셋 내 전체 이미지들 중 해당 개수의 이미지들만을 랜덤하게 샘플링하여 이를 메모리에 로드하도록 합니다. 당연하게도 전체 데이터셋을 학습에 사용하는 경우보다는 최종 테스트 성능이 하락할 것이나, 전체 구현체를 시험적으로 구동해 보는 데에는 크게 문제가 없을 것이라고 생각합니다.

> 하지만 틀림없이 상대적 박탈감을 느끼실 것인데, 이 점에 대해서는 심심한 사과의 말씀을 드립니다.

본래 크기가 큰 데이터셋을 학습 또는 예측 과정에서 미니배치 단위로 불러들이도록 하고자 할 때는, 위와 같은 방식보다는 딥러닝 프레임워크에서 제공하는 input pipeline 관련 API를 사용하여 구현하는 것이 훨씬 효율적입니다. 이번에는 구현체가 지나치게 복잡해지는 것을 방지하고자 위의 방법을 채택하였다는 점을 양지해 주시길 바랍니다.

#### DataSet 클래스

```python
class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C).
        :param labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'Number of examples mismatch, between images and labels.'
            )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: this can be None, if not given.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label indices(can be permuted)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True,
                   fake_data=False):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        :param is_train: bool, current phase for sampling.
        :param fake_data: bool, whether to generate fake data (for debugging).
        :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if fake_data:
            fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
            fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
            fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
            return fake_batch_images, fake_batch_labels

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # Perform data augmentation, for training phase
            batch_images = random_crop_reflect(batch_images, 227)
        elif augment and not is_train:
            # Perform data augmentation, for evaluation phase(10x)
            batch_images = corner_center_crop_reflect(batch_images, 227)
        else:
            # Don't perform data augmentation, generating center-cropped patches
            batch_images = center_crop(batch_images, 227)

        return batch_images, batch_labels
```

데이터셋 요소를 클래스화한 것이 `DataSet` 클래스입니다. 여기에는 기본적으로 이미지들과 이에 해당하는 레이블들이 `np.ndarray` 타입의 멤버로 포함되어 있습니다. 핵심이 되는 부분은 `next_batch` 함수인데, 이는 주어진 `batch_size` 크기의 미니배치(이미지, 레이블)를 현재 데이터셋으로부터 추출하여 반환합니다.

원 AlexNet 논문에서는 학습 단계와 테스트 단계에서의 데이터 증강(data augmentation) 방법을 아래와 같이 서로 다르게 채택하고 있습니다.

- 학습 단계: 원본 $$256\times256$$ 크기의 이미지로부터 $$227\times227$$ 크기의 패치(patch)를 랜덤한 위치에서 추출하고, $$50\%$$ 확률로 해당 패치에 대한 수평 방향으로의 대칭 변환(horizontal reflection)을 수행하여, 이미지 하나 당 하나의 패치를 반환함
- 테스트 단계: 원본 $$256\times256$$ 크기 이미지에서의 좌측 상단, 우측 상단, 좌측 하단, 우측 하단, 중심 위치 각각으로부터 총 5개의 $$227\times227$$ 패치를 추출하고, 이들 각각에 대해 수평 방향 대칭 변환을 수행하여 얻은 5개의 패치를 추가하여, 이미지 하나 당 총 10개의 패치를 반환함

`next_batch` 함수에서는 데이터 증강을 수행하도록 설정되어 있는 경우에 한해(`augment == True`), 현재 학습 단계인지(`is_train == True`) 테스트 단계인지(`is_train == False`)에 따라 위와 같이 서로 다른 데이터 증강 방법을 적용하고, 이를 통해 얻어진 패치 단위의 이미지들을 반환하도록 하였습니다.

### 원 논문과의 차이점

본래 AlexNet 논문에서는 추출되는 패치의 크기가 $$224\times224$$라고 명시되어 있으나, 본 구현체에서는 $$227\times227$$로 하였습니다. 실제로 온라인 상의 많은 AlexNet 구현체에서 $$227\times227$$ 크기를 채택하고 있으며, 이렇게 해야만 올바른 형태로 구현이 가능합니다.

또, AlexNet 논문에서는 여기에 PCA에 기반한 색상 증강(color augmentation)을 추가로 수행하였는데, 본 구현체에서는 구현의 단순화를 위해 이를 반영하지 않았습니다.


## (2) 성능 평가: 정확도

개vs고양이 분류 문제의 성능 평가 척도로는, 가장 단순한 척도인 **정확도(accuracy)**를 사용합니다. 단일 사물 분류 문제의 경우 주어진 이미지를 하나의 클래스로 분류하기만 하면 되기 때문에, 정확도가 가장 직관적인 척도라고 할 수 있습니다. 이는, 테스트를 위해 주어진 전체 이미지 수 대비, 분류 모델이 올바르게 분류한 이미지 수로 정의됩니다.

\begin{equation}
\text{정확도} = \frac{\text{올바르게 분류한 이미지 수}} {\text{전체 이미지 수}}
\end{equation}

### learning.evaluators 모듈

`learning.evaluators` 모듈은, 현재까지 학습된 모델의 성능 평가를 위한 'evaluator(성능 평가를 수행하는 개체)'의 클래스를 담고 있습니다. 

#### Evaluator 클래스

```python
class Evaluator(object):
    """Base class for evaluation functions."""

    @abstractproperty
    def worst_score(self):
        """
        The worst performance score.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        The mode for performance score, either 'max' or 'min'.
        e.g. 'max' for accuracy, AUC, precision and recall,
              and 'min' for error rate, FNR and FPR.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: float, current performance to be evaluated.
        :param best: float, current best performance.
        :return bool.
        """
        pass
```

`Evaluator` 클래스는, evaluator를 서술하는 베이스 클래스입니다. 이는 `worst_score`, `mode` 프로퍼티(property)와 `score`, `is_better` 함수로 구성되어 있습니다. 성능 평가 척도에 따라 '최저' 성능 점수와 '점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지' 등이 다르기 때문에, 이들을 명시하는 부분이 각각 `worst_score`와 `mode`입니다. 

한편 `score` 함수는 테스트용 데이터셋의 실제 레이블 및 이에 대한 모델의 예측 결과를 받아, 지정한 성능 평가 척도에 의거하여 성능 점수를 계산하여 반환합니다. `is_better` 함수는 현재의 평가 성능과 현재까지의 '최고' 성능을 서로 비교하여, 현재 성능이 최고 성능보다 더 우수한지 여부를 `bool` 타입으로 반환합니다.

#### AccuracyEvaluator 클래스

```python
class AccuracyEvaluator(Evaluator):
    """Evaluator with accuracy metric."""

    @property
    def worst_score(self):
        """The worst performance score."""
        return 0.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'max'

    def score(self, y_true, y_pred):
        """Compute accuracy for a given prediction."""
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance score is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
```

`AccuracyEvaluator` 클래스는 정확도를 평가 척도로 삼는 evaluator로, `Evaluator` 클래스를 구현(implement)한 것입니다. `score` 함수에서 정확도를 계산하기 위해, scikit-learn 라이브러리에서 제공하는 `sklearn.metrics.accuracy_score` 함수를 불러와 사용하였습니다. 한편 `is_better` 함수에서는 두 성능 간의 단순 비교를 수행하는 것이 아니라, 상대적 문턱값(relative threshold)를 사용하여 현재 평가 성능이 최고 평가 성능보다 지정한 비율 이상으로 높은 경우에 한해 `True`를 반환하도록 하였습니다.


## (3) 러닝 모델: AlexNet

러닝 모델로는 앞서 언급한 대로 컨볼루션 신경망인 AlexNet을 사용합니다. 이 때, 러닝 모델을 사후적으로 수정하거나 혹은 새로운 구조의 러닝 모델을 추가하는 상황에서의 편의를 고려하여, 컨볼루션 신경망에서 주로 사용하는 층(layers)들을 생성하는 함수를 미리 정의해 놓고, 일반적인 컨볼루션 신경망 모델을 표현하는 베이스 클래스를 먼저 정의한 뒤 이를 AlexNet의 클래스가 상속받는 형태로 구현하였습니다.

### models.layers 모듈

`models.layers` 모듈에서는, 컨볼루션 신경망에서 주로 사용하는 컨볼루션 층(convolutional layer), 완전 연결 층(fully-connected layer) 등을 함수 형태로 정의하였습니다. 

```python
def weight_variable(shape, stddev=0.01):
    """
    Initialize a weight variable with given shape,
    by sampling randomly from Normal(0.0, stddev^2).
    :param shape: list(int).
    :param stddev: float, standard deviation of Normal distribution for weights.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    Initialize a bias variable with given shape,
    with given constant value.
    :param shape: list(int).
    :param value: float, initial value for biases.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    Compute a 2D convolution from given input and filter weights.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    Add a new convolutional layer.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the filters for each dimension.
    :param stride: int, the stride of the filters for each dimension.
    :param out_depth: int, the total number of filters to be applied.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    Add a new fully-connected layer.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, the dimension of output vector.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases
```

AlexNet의 경우 처음 가중치(weight)와 바이어스(bias)를 초기화(initialize)할 때 각기 다른 방법으로 합니다.

- 가중치: 지정한 표준편차(standard deviation)를 가지는 정규 분포(Normal distribution)으로부터 가중치들을 랜덤하게 샘플링하여 초기화함
- 바이어스: 지정한 값으로 초기화함

이를 반영하고자 `weight_variable` 함수에서는 가중치를 샘플링할 정규 분포의 표준편차인 `stddev`을, `bias_variable` 함수에서는 바이어스를 초기화할 값인 `value`를 인자로 추가하였습니다. AlexNet의 각 층에 따라 초기화에 사용할 가중치의 표준편차 및 바이어스 값 등이 다르게 적용되기 때문에, 이를 조정할 수 있도록 구현하였습니다.


### models.nn 모듈

`models.nn` 모듈은, 컨볼루션 신경망을 표현하는 클래스를 담고 있습니다.

#### ConvNet 클래스

```python
class ConvNet(object):
    """Base class for Convolutional Neural Networks."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: tuple, the shape of inputs (H, W, C), ranged [0.0, 1.0].
        :param num_classes: int, the number of classes.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # Build model and loss function
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
            - batch_size: int, batch size for each iteration.
            - augment_pred: bool, whether to perform augmentation for prediction.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start evaluation loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # If performing augmentation during prediction,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # compute predictions for each of 10 patch modes,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # and average predictions on the 10 patches
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # Compute predictions
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred
```

`ConvNet` 클래스는, 컨볼루션 신경망 모델 객체를 서술하는 베이스 클래스입니다. 어떤 컨볼루션 신경망을 사용할 것이냐에 따라 그 **아키텍처(architecture)**가 달라질 것이기 때문에, `ConvNet` 클래스의 자식 클래스에서 이를 `_build_model` 함수에서 구현하도록 하였습니다. 한편 컨볼루션 신경망을 학습할 시 사용할 **손실 함수(loss function)** 또한 `ConvNet`의 자식 클래스에서 `_build_loss` 함수에 구현하도록 하였습니다.

`predict` 함수는, `DataSet` 객체인 `dataset`을 입력받아 이에 대한 모델의 예측 결과를 반환합니다. 이 때, 테스트 단계에서의 데이터 증강 방법을 채택할 경우(`augment_pred == True`), 앞서 설명했던 방식대로 하나의 이미지 당 총 10개의 패치를 얻으며, 이들 각각에 대한 예측 결과를 계산하고 이들의 평균을 계산하는 방식으로 최종적인 예측을 수행하게 됩니다.

#### AlexNet 클래스

```python
class AlexNet(ConvNet):
    """AlexNet class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building AlexNet.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
            - dropout_prob: float, the probability of dropping out each unit in FC layer.
        :return d: dict, containing outputs on each layer.
        """
        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # The probability of keeping each unit for dropout layers
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.)

        # input
        X_input = self.X - X_mean    # perform mean subtraction

        # conv1 - relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = tf.nn.relu(d['conv1'])
        # (227, 227, 3) --> (55, 55, 96)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (55, 55, 96) --> (27, 27, 96)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = tf.nn.relu(d['conv2'])
        # (27, 27, 96) --> (27, 27, 256)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (27, 27, 256) --> (13, 13, 256)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = tf.nn.relu(d['conv3'])
        # (13, 13, 256) --> (13, 13, 384)

        # conv4 - relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv4.shape', d['conv4'].get_shape().as_list())
        d['relu4'] = tf.nn.relu(d['conv4'])
        # (13, 13, 384) --> (13, 13, 384)

        # conv5 - relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv5.shape', d['conv5'].get_shape().as_list())
        d['relu5'] = tf.nn.relu(d['conv5'])
        # (13, 13, 384) --> (13, 13, 256)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (13, 13, 256) --> (6, 6, 256)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        # Flatten feature maps
        f_dim = int(np.prod(d['pool5'].get_shape()[1:]))
        f_emb = tf.reshape(d['pool5'], [-1, f_dim])
        # (6, 6, 256) --> (9216)

        # fc6
        with tf.variable_scope('fc6'):
            d['fc6'] = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu6'] = tf.nn.relu(d['fc6'])
        d['drop6'] = tf.nn.dropout(d['relu6'], keep_prob)
        # (9216) --> (4096)
        print('drop6.shape', d['drop6'].get_shape().as_list())

        # fc7
        with tf.variable_scope('fc7'):
            d['fc7'] = fc_layer(d['drop6'], 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu7'] = tf.nn.relu(d['fc7'])
        d['drop7'] = tf.nn.dropout(d['relu7'], keep_prob)
        # (4096) --> (4096)
        print('drop7.shape', d['drop7'].get_shape().as_list())

        # fc8
        with tf.variable_scope('fc8'):
            d['logits'] = fc_layer(d['relu7'], num_classes,
                                weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments for regularization term.
            - weight_decay: float, L2 weight decay regularization coefficient.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss
```

`AlexNet` 클래스는, AlexNet의 아키텍처 및 학습에 사용할 손실 함수를 정의하고자 `ConvNet` 클래스를 상속받은 것입니다. `_build_model` 함수에서는 전체 아키텍처뿐만 아니라 각 층별 가중치 및 바이어스 초기화를 위한 하이퍼파라미터(hyperparameters; `weights_stddev`, `biases_value`)와, 핵심적인 정규화 기법인 **드롭아웃(dropout)**을 수행하는 부분을 하나의 독립적인 층 형태로 삽입하여 구현하였습니다.

`_build_loss` 함수에서는 AlexNet을 학습하는 데 사용할 **소프트맥스 교차 엔트로피(softmax cross-entropy)** 손실 함수를 구현하였습니다. 이 때, 주요 정규화 기법인 L2 정규화(L2 regularization)를 위해 전체 가중치 및 바이어스에 대한 L2 norm을 계산하고, 여기에 `weight_decay` 인자를 통해 전달된 계수를 곱한 뒤 기존 손실함수에 더하여 최종적인 손실 함수를 완성하였습니다.

### 원 논문과의 차이점

AlexNet 논문에서는, AlexNet의 아키텍처를 아래 그림과 같이 표현하고 있습니다.

{% include image.html name=page.name file="alexnet-architecture.svg" description="AlexNet 아키텍처" class="full-image" %}

중간 컨볼루션 층에서 생성된 3차원 출력값들이 크게 두 갈래로 나뉘어 다음 층으로 전달되고 있는 것을 확인할 수 있는데, 이는 AlexNet 초기 구현 당시 두 개의 GPU를 병렬적으로 활용하기 위해 채택한 **그룹 컨볼루션(grouped convolution)**으로 인한 결과물이라고 할 수 있습니다. 오늘날에는 초기 구현 당시보다 GPU의 성능도 향상된 것도 있고, 구조적으로 봐도 일반적인 형태의 컨볼루션을 채택하는 것이 더 단순하면서 효과적이기 때문에, 본 구현체에서는 그룹 컨볼루션 대신 일반적인 형태의 컨볼루션을 사용하여 전체 아키텍처를 구현하였습니다. 

또, AlexNet 논문에서는 local response normalization(이하 LRN) 층을 몇몇 컨볼루션 층 및 풀링(pooling) 층 바로 다음 위치에 삽입하여, 각 층에서의 출력값을 일정하게 조절하고 있습니다. 오늘날 활성함수(activation function)의 개선 및 batch normalization 방법의 등장으로 인해 LRN은 최신 컨볼루션 신경망 아키텍처에서 거의 사용되지 않으며, 구현의 단순화 측면에서도 좋지 못하기 때문에, 본 구현체에서는 이를 삽입하지 않았습니다.

그리고 AlexNet 논문에서는 몇몇 컨볼루션 층과 완전 연결 층의 바이어스를 1.0으로 초기화하였으나, 본 구현체에서는 이를 1.0 대신 0.1로 초기화하였습니다.


## (4) 러닝 알고리즘: SGD+Momentum

러닝 알고리즘으로는, AlexNet 논문을 그대로 따라하여 **모멘텀(momentum)**을 적용한 **확률적 경사 하강법(stochastic gradient descent; 이하 SGD)**을 채택하였습니다. 이 때, 기존 러닝 알고리즘을 사후적으로 수정하거나 혹은 새로운 러닝 알고리즘을 추가하는 상황에서의 편의를 고려하여, SGD 계열의 러닝 알고리즘에 기반한 'optimizer(최적화를 수행하는 개체)'를 표현하는 베이스 클래스를 먼저 정의한 뒤, 이를 모멘텀 SGD에 기반한 optimizer 클래스가 상속받는 형태로 구현하였습니다. 

### learning.optimizers 모듈

`learning.optimizers` 모듈에서는, optimizer의 베이스 클래스인 `Optimizer` 클래스와 이를 상속받는 `MomentumOptimizer` 클래스를 담고 있습니다.

#### Optimizer 클래스

```python
class Optimizer(object):
    """Base class for gradient-based optimization algorithms."""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet, the model to be learned.
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - batch_size: int, batch size for each iteration.
            - num_epochs: int, total number of epochs for training.
            - init_learning_rate: float, initial learning rate.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # Training hyperparameters
        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)    # Placeholder for current learning rate
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1
        self.num_bad_epochs = 0    # number of bad epochs, where the model is updated without improvement.
        self.best_score = self.evaluator.worst_score    # initialize best score with the worst one
        self.curr_learning_rate = self.init_learning_rate    # current learning rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        tf.train.Optimizer.minimize Op for a gradient update.
        This should be implemented, and should not be called manually.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate (if needed) on every epoch, by its own schedule.
        This should be implemented, and should not be called manually.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        Make a single gradient update and return its results.
        This should not be called manually.
        :param sess: tf.Session.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - augment_train: bool, whether to perform augmentation for training.
        :return loss: float, loss value for the single iteration step.
                y_true: np.ndarray, true label from the training set.
                y_pred: np.ndarray, predicted label from the model.
        """
        augment_train = kwargs.pop('augment_train', True)

        # Sample a single batch
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        # Compute the loss and make update
        _, loss, y_pred = \
            sess.run([self.optimize, self.model.loss, self.model.pred],
                     feed_dict={self.model.X: X, self.model.y: y_true,
                                self.model.is_train: True,
                                self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param sess: tf.Session.
        :param save_dir: str, the directory to save the learned weights of the model.
        :param details: bool, whether to return detailed results.
        :param verbose: bool, whether to print details during training.
        :param kwargs: dict, extra arguments containing training hyperparameters.
        :return train_results: dict, containing detailed results of training.
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())    # initialize all weights

        train_results = dict()    # dictionary to contain training(, evaluation) results and details
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # Start training loop
        for i in range(num_steps):
            # Perform a gradient update from a single minibatch
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            # Perform evaluation in the end of each epoch
            if (i+1) % num_steps_per_epoch == 0:
                # Evaluate model with current minibatch, from training set
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # If validation set is initially given, use it for evaluation
                if self.val_set is not None:
                    # Evaluate model with the validation set
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                # else, just use results from current minibatch for evaluation
                else:
                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                # Keep track of the current best model,
                # by comparing current score and the best score
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))    # save current weights
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results
```

`Optimizer` 클래스는 학습을 통한 모델의 최적화를 담당하기 때문에, 학습 데이터셋(`DataSet` 객체)과 학습할 모델(`ConvNet` 객체), evaluator(`Evaluator` 객체), 그리고 학습과 관련된 기본적인 하이퍼파라미터(`batch_size`, `num_epochs`, `init_learning_rate` 등)들을 멤버 변수 형태로 포함하고 있습니다. SGD 계열의 러닝 알고리즘들 중 구체적으로 어떤 것을 사용할 것인지에 따라, `_optimize_op` 함수에서 이를 명시하여 구현하도록 하였습니다. 

또한 학습 과정에서 모델 업데이트를 거듭할수록 매 epochs의 말미마다 **학습률(learning rate)**을 낮춰주어 모델이 수렴(convergence)하도록 유도할 수 있는데, 이를 위한 학습률 스케줄링(scheduling) 방법을 `_update_learning_rate` 함수에서 구현하도록 하였습니다. 이 함수에서 현재의 학습률을 나타내는 `curr_learning_rate` 멤버 변수의 값을 일정한 규칙에 의거하여 조정하도록 할 수 있습니다.

`train` 함수는 `Optimizer` 클래스의 핵심적인 함수로, 실제 학습을 수행하도록 합니다. `batch_size` 크기의 미니배치를 학습 데이터셋으로부터 추출하여 이에 대한 손실 함수의 값을 계산하고, 이를 사용하여 SGD에 기반한 모델 파라미터의 업데이트를 수행합니다. 이 과정에서 매 epoch의 말미에 도달할 때마다, 검증 데이터셋(validation set)에 대하여 현재까지 학습된 모델의 성능을 평가합니다(만약 검증 데이터셋이 따로 주어지지 않은 경우, 기존 학습 데이터셋의 해당 반복 회차에서 추출된 미니배치에 대한 성능 평가로 대체합니다). 현재 모델의 성능 평가 결과가 (`Evaluator`의 `is_better` 함수에 의해) 지금까지의 최고 성능 기록보다 높다고 판단될 경우, 현재 모델의 파라미터를 디스크에 저장하고 최고 성능 기록을 업데이트합니다. 

#### MomentumOptimizer 클래스

```python
class MomentumOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _optimize_op(self, **kwargs):
        """
        tf.train.MomentumOptimizer.minimize Op for a gradient update.
        :param kwargs: dict, extra arguments for optimizer.
            - momentum: float, the momentum coefficient.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate, when evaluation score plateaus.
        :param kwargs: dict, extra arguments for learning rate scheduling.
            - learning_rate_patience: int, number of epochs with no improvement
                                      after which learning rate will be reduced.
            - learning_rate_decay: float, factor by which the learning rate will be updated.
            - eps: float, if the difference between new and old learning rate is smaller than eps,
                   the update is ignored.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # Decay learning rate only when the difference is higher than epsilon.
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
```

`MomentumOptimizer` 클래스는, SGD+Momentum 기반의 optimizer를 정의하고자 `Optimizer` 클래스를 상속받은 것입니다. `_optimize_op` 함수에서는 모멘텀의 정도를 조절하는 계수를 `momentum` 인자로 전달받아, TensorFlow에서 제공하는 `tf.train.MomentumOptimizer` 에 대한 `minimize` Operation을 반환합니다.

`_update_learning_rate` 함수에서는, AlexNet 논문에서 채택한 아래의 학습률 스케줄링 방법을 좀 더 고도화하여 구현하였습니다.

- 현재 수준의 학습률에서, 검증 데이터셋에 대한 성능 향상이 수 epochs에 걸쳐 확인되지 않을 경우, 학습률을 일정한 수로 나누어 감소시킨 뒤 학습을 지속함

`learning_rate_patience`는, 몇 epochs동안 성능 향상이 확인되지 않을 경우 학습률을 조정할지 결정하는 인자이며, `learning_rate_decay`의 경우, 학습률 조정 시 곱하여 학습률을 감소시키기 위해 사용되는 값을 전달하는 인자입니다. 학습률을 일정 비율로 계속 감소시키다 보면 어느 순간부터는 학습률의 변화량이 미미해지는데, 이 때 이전 학습률과 조정된 학습률 간의 차이가 `eps` 인자의 값보다 큰 경우에 한해서만 학습률 조정을 실제로 수행하도록 합니다.


## 학습 수행 및 테스트 결과

`train.py` 스크립트에서는 실제 학습을 수행하는 과정을 구현하였으며, `test.py` 스크립트에서는 테스트 데이터셋에 대하여 학습이 완료된 모델을 테스트하는 과정을 구현하였습니다. 

### train.py 스크립트

```python
import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator


""" 1. Load and split datasets """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# Load trainval set and split into train/val sets
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# Sanity check
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. Set training hyperparameters """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # mean image
np.save('/tmp/asirra_mean.npy', image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Training hyperparameters
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 300

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4


""" 3. Build graph, initialize a session and start training """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
```

`train.py` 스크립트에서는 다음의 3단계 과정을 거칩니다.

1. 원본 학습 데이터셋을 메모리에 로드하고, 이를 학습 데이터셋(80%)과 검증 데이터셋(20%)으로 나눈 뒤 각각을 사용하여 `DataSet` 객체를 생성함 
2. 학습 수행 및 성능 평가와 관련된 하이퍼파라미터를 설정함
3. `ConvNet` 객체, `Evaluator` 객체 및 `Optimizer` 객체를 생성하고, TensorFlow Graph와 Session을 초기화한 뒤, `Optimizer.train` 함수를 호출하여 모델 학습을 수행함

이 때, 원본 데이터셋 저장 경로, 하이퍼파라미터 등 `FIXME`로 표시된 부분은 여러분의 상황에 맞춰 수정하셔야 합니다. 본 글에서 학습을 수행할 당시의 하이퍼파라미터 설정은 아래와 같이 하였습니다.

- 러닝 알고리즘 관련 하이퍼파라미터 설정
  - Batch size: 256
  - Number of epochs: 300
  - Initial learning rate: 0.01
  - Momentum: 0.9
  - Learning rate decay: 0.1
    - Learning rate patience: 30
    - eps: 1e-8
- 정규화 관련 하이퍼파라미터 설정
  - L2 weight decay: 0.0005
  - dropout probability: 0.5
- 평가 척도 관련 하이퍼파라미터 설정
  - Score threshold: 1e-4

### test.py 스크립트

```python
""" 1. Load and split datasets """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# Load test set
X_test, y_test = dataset.read_asirra_subset(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

# Sanity check
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. Set test hyperparameters """
hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')    # load mean image
hp_d['image_mean'] = image_mean

# FIXME: Test hyperparameters
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/tmp/model.ckpt')    # restore learned weights
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
```

`test.py` 스크립트에서도 유사하게, 다음의 3단계 과정을 거칩니다. 

1. 원본 테스트 데이터셋을 메모리에 로드하여, `DataSet` 객체를 생성함
2. 테스트 수행 및 성능 평가와 관련된 하이퍼파라미터를 설정함
3. `ConvNet` 객체와 `Evaluator` 객체를 생성하고, TensorFlow Graph와 Session을 초기화한 뒤, `tf.train.Saver`의 `restore` 함수 호출을 통해 학습이 완료된 모델의 가중치들을 로드하고 `ConvNet.predict` 함수와 `Evaluator.score` 함수를 순서대로 호출하여 모델 테스트 성능을 평가함

### 학습 결과 분석

#### 학습 곡선

`train.py` 스크립트를 실행하여, 실제 학습 수행 과정에서 아래의 정보들을 추적하여, 이를 **학습 곡선(learning curve)**으로 나타내었습니다.

- 매 반복 횟수에서의 손실 함수의 값
- 매 epoch에 대하여 (1) 학습 데이터셋으로부터 추출한 미니배치에 대한 모델의 예측 정확도(이하 학습 정확도)와 (2) 검증 데이터셋에 대한 모델의 예측 정확도(이하 검증 정확도)

{% include image.html name=page.name file="learning-curve-result.svg" description="학습 곡선 플롯팅 결과" class="large-image" %}

학습이 진행됨에 따라 손실 함수의 값은 (약간의 진동이 있으나) 점차적으로 감소하면서, 동시에 학습 정확도 및 검증 정확도는 점차적으로 증가하는, 꽤 예쁜(?) 결과를 보였습니다. 단, 학습 정확도가 1.0을 향해 가는 과정에서 검증 정확도는 약 0.9288 언저리에 머물렀는데, 이 차이는 모델이 학습 데이터셋에 대하여 과적합(overfitting)된 정도를 나타낸다고 할 수 있겠습니다. 

학습 과정 말미에서, 검증 정확도가 0.932일 때의 모델 가중치를 최종적으로 채택하여, 테스트를 위해 저장하였습니다.

#### 테스트 결과

테스트 결과 측정된 정확도는 **0.92768**로 확인되었습니다. <a href="https://www.kaggle.com/c/dogs-vs-cats" target="_blank">Dogs vs. Cats</a>의 Leaderboard 섹션에서, 1등인 Pierre Sermanet이 거둔 0.98914에 비하면 한참 못 미치는 점수입니다. 그러나 (1) 원본 데이터셋(25,000장)의 절반(12,500장)밖에 학습에 사용하지 않았으며, (2) 단 한 개의 (튜닝을 거치지 않은) 순수한 AlexNet만을 사용했다는 것을 생각해보면, 필자 생각에는 그렇게 나쁜 결과도 아닌 것 같습니다.

실제로 얻을 수 있는 이미지에 대한 테스트를 위해, Google에서 개 이미지와 고양이 이미지 각각에 대한 검색 결과들 중 랜덤하게 3개씩 고른 뒤 이들을 학습이 완료된 모델에 입력하였더니, 아래와 같은 예측 결과를 얻었습니다. 

{% include image.html name=page.name file="random-dogs-cats-predictions.png" description="랜덤한 개vs고양이 이미지에 대한 모델의 예측 결과(pred)" class="full-image" %}


## 결론

본 글에서는 이미지 인식 분야에서 가장 많이 다뤄지는 Classification 문제의 예시로, '개vs고양이 분류' 문제를 정하고, 이를 AlexNet 모델과 딥러닝 알고리즘을 사용하여 해결하는 과정을 안내하였습니다. 비록 온라인 상에서 딥러닝 구현체를 쉽게 찾을 수 있더라도, 여러분들이 데이터셋, 성능 평가, 러닝 모델, 러닝 알고리즘의 4가지 요소를 고려하여 각각을 모듈화하는 방식으로 직접 구현한다면, 딥러닝에 대한 이해 및 구현체에 대한 유지/보수의 측면에서 장점을 가져다줄 수 있다고 말씀드렸습니다. 여러분들이 본 글에서 제공한 구현체를 보고 받아들이는 입장에서도, 이러한 장점을 어느 정도는 느낄 수 있으셨기를 바랍니다. 

\*추후 글에서는, 이미지 인식 분야의 또 다른 중요한 문제들인 Detection 및 Segmentation 문제를 해결하는 과정을, 예시 문제와 모델 등을 선정하여 여러분들께 안내해 드리고자 합니다. 이 때에도 본 글에서 언급한 딥러닝의 4가지 기본 요소를 중심으로 할 것입니다. 


## References

- AlexNet 논문
  - <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" target="_blank">Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.</a>
- The Asirra dataset
  - <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/CCS2007.pdf" target="_blank">Elson, Jeremy, et al. "Asirra: a CAPTCHA that exploits interest-aligned manual image categorization." (2007).</a>
- 본문 구현체의 러닝 알고리즘의 학습률 스케줄링 방법
	- <a href="http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau" target="_blank">"torch.optim.lr_scheduler.ReduceLROnPlateau" Pytorch master documentation, http://pytorch.org/docs/master/optim.html. Accessed 4 January 2018.</a>
