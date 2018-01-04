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

머신러닝의 핵심 요소를 고려하여, 딥러닝 구현체를 위와 같이 총 4가지 기본 요소로 구분지어 이해하고자 시도한다면, 딥러닝 관련 개념들을 이해하고 이를 기반으로 자신만의 구현체를 만드는 데 매우 유용합니다. 뿐만 아니라, 새로운 딥러닝 관련 테크닉이 나오게 되더라도 그것이 위 4가지 요소 중 어느 부분에 해당하는 것인지를 먼저 파악하게 되면, 그것을 구현하는 시간을 그만큼 단축시킬 수 있습니다. 예를 들어 'DenseNet은 기존 모델에 skip connections를 무수히 많이 추가한 것이므로, 기존 러닝 모델에 이를 추가하면 되겠구나!' 내지는 'Adam은 기존의 SGD에 adaptive moment estimation을 추가한 것이므로, 기존 러닝 알고리즘에 이를 추가하면 되겠구나!' 하는 식으로 생각할 수 있을 것입니다.

그러면 지금부터, 위에서 언급한 4가지 딥러닝 요소를 기준으로 하나씩 살펴보면서, '개vs고양이 분류' 문제를 AlexNet을 사용하여 해결해보도록 하겠습니다.


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

이 때 주의할 점은, 위 함수를 사용하여 전체 학습 데이터셋을 메모리에 로드하고자 할 경우, *적어도 16GB 이상의 메모리를 필요로 한다*는 점입니다. 메모리 사양이 이에 못 미치는 분들을 위한 궁여지책(?)으로 함수에 `sample_size` 인자를 추가하였는데, 이를 명시할 경우 원본 데이터셋 내 전체 이미지들 중 해당 개수의 이미지들만을 랜덤하게 샘플링하여 이를 메모리에 로드하도록 합니다. 당연하게도 전체 데이터셋을 학습에 사용하는 경우보다는 최종 테스트 성능이 하락할 것이나, 전체 구현체를 시험적으로 구동하는 데에는 크게 문제가 없을 것이라고 생각합니다.

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

- 학습 단계: 원본 $$256\times256$$ 크기의 이미지로부터 $$224\times224$$ 크기의 패치(patch)를 랜덤한 위치에서 추출하고, $$50%$$ 확률로 해당 패치에 대한 수평 방향으로의 대칭 변환(horizontal reflection)을 수행하여, 이미지 하나 당 하나의 패치를 반환함
- 테스트 단계: 원본 $$256\times256$$ 크기 이미지에서의 좌측 상단, 우측 상단, 좌측 하단, 우측 하단, 중심 위치 각각으로부터 총 5개의 패치를 추출하고, 이들 각각에 대해 수평 방향 대칭 변환을 수행하여 얻은 5개의 패치를 추가하여, 이미지 하나 당 총 10개의 패치를 반환함

`next_batch` 함수에서는 데이터 증강을 수행하도록 설정되어 있는 경우에 한해(`augment == True`), 현재 학습 단계인지(`is_train == True`) 테스트 단계인지(`is_train == False`)에 따라 위와 같이 서로 다른 데이터 증강 방법을 적용하고, 이를 통해 얻어진 패치 단위의 이미지들을 반환하도록 하였습니다.

단, 원 AlexNet 논문에서는 여기에 PCA에 기반한 색상 증강(color augmentation)을 추가로 수행하였는데, 본 구현체에서는 구현의 단순화를 위해 이를 반영하지 않았습니다.


## (2) 성능 평가: 정확도

개vs고양이 분류 문제의 성능 평가 척도로는, 가장 단순한 척도인 **정확도(accuracy)**를 사용합니다. 단일 사물 분류 문제의 경우 주어진 이미지를 하나의 클래스로 분류하기만 하면 되기 때문에, 정확도가 가장 직관적인 척도라고 할 수 있습니다. 이는, 테스트를 위해 주어진 전체 이미지 수 대비, 분류 모델이 올바르게 분류한 이미지 수로 정의됩니다.

\begin{equation}
\text{정확도} = \frac{\text{올바르게 분류한 이미지 수}} {\text{전체 이미지 수}}
\end{equation}

### learning.evaluators 모듈

`learning.evaluators` 모듈은, 현재까지 학습된 모델의 성능 평가를 위한 클래스를 담고 있습니다. 

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

`Evaluator` 클래스는, 성능 평가를 담당하는 객체를 서술하는 추상(abstract) 클래스입니다. 이는 `worst_score`, `mode` 프로퍼티(property)와 `score`, `is_better` 함수로 구성되어 있습니다. 성능 평가 척도에 따라 '최저' 성능 점수와 '점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지' 등이 다르기 때문에, 이들을 명시하는 부분이 각각 `worst_score`와 `mode`입니다. 

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

`AccuracyEvaluator` 클래스는 정확도를 평가 척도로 삼는 것으로, `Evaluator` 클래스를 구현(implement)한 것입니다. `score` 함수에서 정확도를 계산하기 위해, scikit-learn 라이브러리에서 제공하는 `sklearn.metrics.accuracy_score` 함수를 불러와 사용하였습니다. 한편 `is_better` 함수에서는 두 성능 간의 단순 비교를 수행하는 것이 아니라, 상대적 문턱값(relative threshold)를 사용하여 현재 평가 성능이 최고 평가 성능보다 지정한 비율 이상으로 높은 경우에 한해 `True`를 반환하도록 하였습니다.


## (3) 러닝 모델: AlexNet

### models.layers 모듈

```python
def weight_variable(shape, stddev=0.01):
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases
```

### models.nn 모듈

#### ConvNet 클래스

```python
class ConvNet(object):
    """
    Base class for Convolutional Neural Networks.
    """

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: Tuple, shape of inputs (H, W, C), range [0.0, 1.0].
        :param num_classes: Integer, number of classes.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])

        self.is_train = tf.placeholder(tf.bool)
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Model builder.
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
        :param verbose: Boolean, whether to print details during prediction.
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Evaluation loop
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
            # else:  X.shape: (N, h, w, C)

            # If performing augmentation during prediction,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, 2), dtype=np.float32)    # (N, 10, 2)
                # compute predictions for each of 10 patch modes,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # and average predictions on the 10 patches
                y_pred = y_pred_patches.mean(axis=1)    # (N, 2)
            else:
                # Compute predictions
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})
                # (N, 2)

            _y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, 2)

        return _y_pred
```

#### AlexNet 클래스

```python
class AlexNet(ConvNet):
    """AlexNet class."""

    def _build_model(self, **kwargs):
        """Model builder."""
        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # keep_prob for dropout layers
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

        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """Evaluate loss for the model."""
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss
```

- 기존 본문과의 차이점
  - 입력층의 크기로 $$224\times224\times3$$ 대신 $$227\times227\times3$$을 사용
  - 그룹 컨볼루션(grouped convolution) 대신 일반적인 형태의 컨볼루션으로 구현
  - Local response normalization 층을 제거함
  - Weight initialization 수행 시, bias 초깃값을 1.0 대신 0.1로 사용


## (4) 러닝 알고리즘: SGD+Momentum

### learning.optimizers 모듈

#### Optimizer 클래스

```python
class Optimizer(object):
    """
    Base class for gradient-based optimization functions.
    """

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: Model to be learned.
        :param train_set: DataSet.
        :param evaluator: Evaluator.
        :param val_set: DataSet.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.optimize = self._optimize_op()
        self.saver = tf.train.Saver()

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

    def train(self, sess, details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param sess: tf.Session.
        :param details: Boolean, whether to return detailed results.
        :param verbose: Boolean, whether to print details during training.
        """
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
        for i in range(num_steps):
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            if (i+1) % num_steps_per_epoch == 0:
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)
                if self.val_set is not None:
                    # Evaluate current model
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            plot=False, save=True, img_dir='/tmp')

                    # Keep track of the current best model for validation set
                    if self.evaluator.is_better(eval_score, self.best_score, **kwargs):
                        self.best_score = eval_score
                        self.num_bad_epochs = 0
                        self.saver.save(sess, '/tmp/model.ckpt')    # save current weights
                    else:
                        self.num_bad_epochs += 1

                else:
                    if verbose:
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            plot=False, save=True, img_dir='/tmp')

                    # Keep track of the current best model for training set
                    if self.evaluator.is_better(step_score, self.best_score, **kwargs):
                        self.best_score = step_score
                        self.num_bad_epochs = 0
                        self.saver.save(sess, '/tmp/model.ckpt')    # save current weights
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

            return details
```

#### MomentumOptimizer 클래스

```python
class MomentumOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _optimize_op(self, **kwargs):
        """tf.train.MomentumOptimizer.minimize Op for a gradient update."""
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """Update current learning rate, when evaluation score plateaus."""
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



## 학습 수행 및 결과

- 학습 관련 hyperparameters:
  - batch_size: 256
  - initial learning rate: 0.01
  - momentum: 0.9
  - learning rate decay rate: 0.1
  - number of epochs: 320 *FIXME*
- 정규화 관련 hyperparameters:
  - L2 weight decay: 0.0005
  - dropout probability: 0.5
- Learning curve + overfitting 여부 확인
- Test set 예시 이미지 - 예측 결과


## 결론

TODO


## References

- AlexNet 논문
  - <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" target="_blank">Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.</a>
- The Asirra dataset
  - <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/CCS2007.pdf" target="_blank">Elson, Jeremy, et al. "Asirra: a CAPTCHA that exploits interest-aligned manual image categorization." (2007).</a>

