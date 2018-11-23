---
layout: post
title: "이미지 Segmentation 문제와 딥러닝: GCN으로 개 고양이 분류 및 분할하기"
date: 2018-11-23 09:00:00 +0900
author: jongsoo_keum
categories: [machine-learning, computer-vision]
tags: [segmentation, GCN, tensorflow]
comments: true
name: image-segmentation-deep-learning
---

안녕하세요! 딥러닝을 이용한 Detection 문제 해결 포스팅에 이어, 정말 오랜만에 이미지 인식 **Segmentation**에 관한 문제 해결 사례를 소개해드리도록 하겠습니다. 이번 포스팅도 앞선 포스팅과 마찬가지로 **TensorFlow** 구현 코드와 함께 진행됩니다. Segmentation의 경우 많은 시간 및 메모리를 소모하며 설명자료가 Classification 문제에 비해 많지 않아 어렵게 느껴질 수도 있지만, Segmentation은 **Pixel-wise classification**으로 생각할 수 있기 때문에 구현 자체 난이도는 Classification과 크게 다르지 않습니다. 다소 난해한 Detection에 비해 금방 이해할 수 있으리라 생각하며, 이전 포스팅과 마찬가지로 개념적인 설명은 조금 뒤로하고 **구현 위주**로 설명해 드리도록 하겠습니다. 이해 안 되는 부분은 언제든 댓글 남겨주세요!

* **다음과 같은 사항을 알고 계시면 더 이해하기 쉽습니다.**
  - 딥러닝에 대한 전반적인 이해
  - Python 언어 및 TensorFlow 프레임워크에 대한 이해
* 이번 글에서 구현한 GCN(Global Convolutional Network)의 경우, 논문에서 명시된 것처럼 ResNet을 자체적으로 변형하여 학습한 pre-trained model을 사용하지 않고, 학습방법을 단순화하여 성능이 논문과 상이할 가능성이 있습니다.
* 이번 글에서는 과거 Classification 구현체와 마찬가지로 데이터셋(data set), 성능 평가(performance evaluation), 러닝 모델(learning model), 러닝 알고리즘(learning algorithm) 4가지 요소를 나눠 구현하였으며, 중복을 피하고자 다르게 구현한 부분 위주로 설명합니다. 
  - 전체 구현체 코드는 <a href="https://github.com/sualab/tf-segmentation" target="_blank">수아랩의 GitHub 저장소</a>에서 자유롭게 확인하실 수 있습니다.
  - 본 글에서 사용한 개vs고양이 데이터셋은 <a href="http://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank">이곳</a>에서 다운로드 받으실 수 있습니다. 참고로, 받은 데이터셋이 데이터 로드 중에 깨지거나 너무 크기가 큰 이미지들이 존재하여, 이를 제거하고 사용하였습니다.
  - 제가 전처리한 데이터셋은 <a href="https://drive.google.com/file/d/1SD30E3Fj3216kHy_k71r5g_AG_uF1HRI/view?usp=sharing" target="_blank">여기</a>서 받을 수 있습니다. 

## 서론

<a href="{{ site.url }}/computer-vision/2017/11/29/image-recognition-overview-1.html" target="_blank">이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로</a>에서 언급한 바와 같이, **PASCAL VOC challenge**에서 중요하게 다루는 3가지 이미지 인식 문제 중 Classification, Detection 이어서 마지막으로 **Segmentation** 기술로 해결할 수 있는 간단한 사례를 소개하고, 이를 딥러닝 기술 중 **Encoder-Decoder 구조** 중 널리 알려진 **GCN(Global Convolutional Network)** 망을 통해 해결하는 과정을 설명드리겠습니다.

앞서 말씀드린 것과 같이 이번 포스팅은 개념적인 설명부터 시작하면 지나치게 글이 길어지고 집중도가 떨어질 것 같아 Detection 포스팅과 마찬가지로 (1) 딥러닝 및 머신러닝에 대한 어느정도의 이해와, (2) Python 언어 및 TensorFlow 프레임워크에 대한 이해, 그리고 (3) Semantic Segmentation 알고리즘에 대한 전반적인 이해를 알고 있다는 전제하에 글을 쓰겠습니다. 글을 읽다가 이해가 안되는 부분이 있다면 다른곳에 좋은 글이 많이 있으니 숙지하신 후 읽어보시길 권장해 드립니다.

본 포스팅에서 해결할 Segmentation 문제는 Classification때와 비슷하게 개와 고양이를 분류, 분할하는 문제로 선택하였습니다. 뒤에서 살펴보겠지만, 한 이미지에 단 하나의 개 혹은 고양이 하나의 객체만 존재하기 때문에, **Cityscape** 혹은 **PASCAL VOC challenge** 보다 해결할 문제가 훨씬 쉬운 편입니다. 하지만, 즉각적으로 구현한 모델이 어느 정도 정상적으로 작동하는지 확인하기 좋고 구현을 통해 Segmentation 문제를 이해하는 데 부족함이 없어 채택하였습니다.

**개 vs 고양이 분할** 문제를 해결하기 위한 딥러닝 알고리즘으로는 **Encoder-Decoder**계열의 **GCN**(**Global Convolutional Network**)를 채택하였습니다. 공개된 Segmentation(TensorFlow) 구현체의 경우, DeepLab으로 대표되는 **Dilated convolution(Atrous convolution)** 계열이 대부분입니다. Dilated convolution 계열의 경우 성능적으로 우수하나, 많은 메모리를 잡아먹고 속도 면에서 좋지 않은 경우가 많습니다. 추가로, DeepLab 계열은 TensorFlow 자체에 구현체가 있으므로 공부하기가 편하나, Encoder-Decoder 계열은 상대적으로 구현체가 적어 차별화(?)를 위해 **GCN**을 선택하였습니다.

GCN 구현체는 앞선 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">Classification 문제</a>와 같이 데이터셋(data set), 성능 평가(performance evaluation), 러닝 모델(learning model), 러닝 알고리즘(learning algorithm) 4가지 요소를 중심으로 작성하였으며, 이전 포스팅과 겹치지 않은 부분 위주로 소개해드리겠습니다.

## (1) 데이터셋: 개vs고양이 

개vs고양이 분류 문제를 위해 사용한 데이터셋은 <a href="http://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank">The Oxford-IIIT Pet Dataset</a>에서 가져왔습니다. 가져온 데이터셋의 정보를 모두 사용하려고 하였으나, 파일을 읽어들일 때 깨지는 경우와, 해상도가 1000 이상으로 큰 경우가 있어 약간의 처리를 거친 후 남은 이미지를 사용하였습니다. (깨진 이미지 제거, 해상도 600 이상 제거)

데이터를 저장해두는 방식은 무수히 많지만, 이번 포스팅 예제에서는 원본을 해치지 않는 선에서 다음과 같은 형식으로 진행하도록 하겠습니다. 
* 이미지(jpg)마다 매칭되는 **mask** 이미지(png)를 가집니다. ex) Bombay_217.jpg <--> Bombay_217.png
* mask 이미지는 세가지 값 (1:Foreground, 2:Background, 3:Unknown)을 가지며, 각 파일명의 영어 부분은 개 혹은 고양의 종 이름을 나타내며, 맨 앞글자가 대문자면 고양이 그렇지 않으면 개를 나타냅니다. 이번 포스팅에서는 크게 **개와 고양이만을 분류, 분할하는 문제**로 간소화 하겠습니다.
* 폴더에 images 폴더, masks 폴더를 두고 images 폴더에 이미지를, masks 폴더에 mask 이미지를 넣어둡니다.

{% include image.html name=page.name file="example.png" description="개vs고양이 데이터셋 예시" class="full-image" %}

데이터셋은 총 4,817장으로 이중 임의로 10%를 골라 480장을 테스트셋으로 사용하였습니다. 클래스는 "개", "고양이" 두 가지로 객체가 있을 곳을 예측하여 배경과 분할하는 문제로 진행됩니다.

### datasets.data 모듈

`datasets.data` 모듈은 데이터셋에 관련된 함수와 클래스를 가지고 있습니다. Classification 문제때와 마찬가지로, 이 모듈은 데이터셋을 메모리에 로드하고 학습 및 예측평가 과정에서 미니배치(minibatch) 단위로 제공해주는 역할을 합니다.

#### read\_data 함수

```python
def read_data(data_dir, image_size, no_label=False):
    """
    Segmentation문제를 위해 데이터를 전처리하고 로드.
    :param data_dir: image 및 mask 데이터가 저장된 root 경로
    :image_size: tuple, 크롭 및 패딩을 위해 지정된 이미지 사이즈
    :no_label: bool, 레이블을 로드할 지 여부
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, H, W, num_classes (include background)).
    """
    im_paths = []
    for x in EXT:
        im_paths.extend(glob.glob(os.path.join(data_dir, 'images', '*.{}'.format(x))))
    imgs = []
    labels = []
    for im_path in im_paths:
        #이미지 로드
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        im = cv2.imread(im_path)
        im = crop_shape(im, image_size)
        im = padding(im, image_size)
        imgs.append(im)
        
        if no_label:
            labels.append(0)
            continue

        #마스크 로드
        mask_path = os.path.join(data_dir, 'masks', '{}.png'.format(im_name))
        mask = cv2.imread(mask_path)
        mask = crop_shape(mask, image_size)
        mask = padding(mask, image_size, fill=2)

        label = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
        label.fill(-1)
        # Pixel annotations 1:Foreground, 2:Background, 3:Unknown
        idx = np.where(mask == 2)
        label[idx[0],idx[1],:] = [1, 0, 0]

        idx = np.where(mask == 1)
        if im_name[0].isupper():
            label[idx[0],idx[1],:] = [0, 1, 0]
        else:
            label[idx[0],idx[1],:] = [0, 0, 1]
        labels.append(label)

    X_set = np.array(imgs, dtype=np.float32)
    y_set = np.array(labels, dtype=np.float32)

    return X_set, y_set
```

`load_data` 함수는 위의 형식으로 저장된 데이터셋을 불러와 각 이미지를 Crop 혹은 Padding을 통해 원하는 크기로 맞춘 뒤, `np.ndarray` 형태로 반환합니다. 마찬가지로 mask 이미지를 불러와 각 픽셀에 대해 one-hot encoding된 label을 `np.ndarray` 형태로 반환합니다. 만약 이미지 사이즈가 $$512 \times 512$$이라면 `y_set`의 형태는 `(N, 512, 512, class 개수 (background 포함))`로 표현됩니다. 

#### DataSet 클래스

```python
class DataSet(object):

    def __init__(self, images, labels=None):
        """
        새로운 DataSet 객체를 생성함.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape: (N, H, W, num_classes (include background)).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0],\
                ('Number of examples mismatch, between images and labels')
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels  # NOTE: this can be None, if not given.
        # image/label indices(can be permuted)
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()

    def _reset(self):
        """일부 변수를 재설정함."""
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

    def sample_batch(self, batch_size, shuffle=True):
        """
        'batch_size' 개수만큼 데이터들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 '한번' 반환함.
        :param batch_size: int,  미니배치 크기
        :param shuffle: bool, 추출 이전에, 데이터셋 이미지를 섞을지 여부
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        if shuffle:
            indices = np.random.choice(self._num_examples, batch_size)
        else:
            indices = np.arange(batch_size)
        batch_images = self._images[indices]
        if self._labels is not None:
            batch_labels = self._labels[indices]
        else:
            batch_labels = None
        return batch_images, batch_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        'batch_size' 개수만큼 데이터들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 반환함.
        :param batch_size: int, 미니배치 크기
        :param shuffle: bool, 추출 이전에, 데이터셋 이미지를 섞을지 여부
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        start_index = self._index_in_epoch

        # 맨 첫 번째 epoch에서 전체 데이터셋을 랜덤하게 섞음
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # 현재의 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch을 진행함
        if start_index + batch_size > self._num_examples:
            # epochs 수를 1 증가
            self._epochs_completed += 1
            # 새로운 epoch에서, 남은 데이터들을 가져옴
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # epoch가 끝나면, 데이터를 섞음
            if shuffle:
                np.random.shuffle(self._indices)

            # 다음 epoch 진행
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self._images[indices_rest_part]
            images_new_part = self._images[indices_new_part]
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)
            if self._labels is not None:
                labels_rest_part = self._labels[indices_rest_part]
                labels_new_part = self._labels[indices_new_part]
                batch_labels = np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self._images[indices]
            if self._labels is not None:
                batch_labels = self._labels[indices]
            else:
                batch_labels = None

        return batch_images, batch_labels
```

Classification 문제때와 마찬가지로 `DataSet` 클래스를 이용하여 메모리에 로드된 `X_set`과 `y_set`을 미니배치(minibatch) 단위로 반환해 줍니다.

### (2) 성능 평가: Pixel Accuracy

모델의 분할 성능 평가를 위해 **Pixel Accuracy**를 사용합니다. Segmentation에서는 mIoU(mean Intersection over union)가 가장 빈번하게 사용되는 성능 척도이나, 이미지에서 객체의 크기가 크고 한가지 밖에 없는 편이라 굳이 IoU까지 사용할 필요가 없다고 판단하여 Pixel Accuracy를 성능 평가 척도로 사용하였습니다. 이는 이미지 단위에서 전체 pixel 개수 대비 올바르게 분류한 pixel의 수로 정의됩니다.

\begin{equation}
\text{Pixel Accuracy} = \frac{\text{올바르게 분류한 Pixel의 수}} {\text{전체 Pixel의 수}}
\end{equation} 

### learning.evaluators 모듈

Classification 문제와 마찬가지로, 성능 평가를 위한 `evaluator` 클래스를 담고 있습니다.

#### Evaluator 클래스

```python
class Evaluator(metaclass=ABCMeta):
    """성능 평가를 위한 evaluator의 베이스 클래스."""

    @abstractproperty
    def worst_score(self):
        """
        최저 성능 점수.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나.
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        실제로 사용할 성능 평가 지표.
        해당 함수를 추후 구현해야 함.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        해당 함수를 추후 구현해야 함.
        :param curr: float, 평가 대상이 되는 현재 성능 점수.
        :param best: float, 현재까지의 최고 성능 점수.
        :return bool.
        """
        pass
```

추상 베이스 클래스입니다. 다른 부분은 모두 Classification 문제에서 서술한 내용과 같아 추가 설명은 생략하겠습니다.

#### AccuracyEvaluator 클래스

```python
class AccuracyEvaluator(Evaluator):
  """ Pixel Accuracy를 성능 평가 척도로 사용하는 evaluator 클래스"""
    @property
    def worst_score(self):
        """최저 성능 점수"""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지 낮아야 우수한지 여부"""
        return 'max'

    def score(self, y_true, y_pred):
        """주어진 예측 마스크 이미지에 대해 Pixel Accuracy를 계산"""
        acc = []
        for t, p in zip(y_true, y_pred):
            # Unknown 영역은 제외하고 pixel accuracy 계산
            ignore = np.where(t[...,0].reshape(-1) == -1)
            acc.append(accuracy_score(np.delete(t.argmax(axis=-1).reshape(-1), ignore[0]),
                                      np.delete(p.argmax(axis=-1).reshape(-1), ignore[0])))
        return sum(acc)/len(acc)

    def is_better(self, curr, best, **kwargs):
        """
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다
        우수한지 여부를 반환하는 함수.
        :param kwargs: dict, 추가 인자.
            - score_threshold: float, 새로운 최적값 결정을 위한 상대적 문턱값으로,
                               유의미한 차이가 발생했을 경우만을 반영하기 위함.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
```

Pixel Accuracy를 성능 평가 척도로 사용하기 위해 상속받아 `AccuracyEvaluator` 클래스를 구현하였습니다. 앞서 설명했듯이, 다루는 데이터셋의 경우 Unknown region이 있기 때문에 Pixel accuracy를 계산할 때 Unknown region은 계산에 포함되지 않도록 정의하고 계산하였습니다. Accuracy는 0.0부터 1.0까지 나타날 수 있으며, 높을 수록 좋은 성능 척도이기 때문에, `mode`를 'max'로, `score_threshold` 값을 1e-4로 설정하였습니다.


## 러닝 모델: GCN(Global Convolutional Network)

러닝 모델로는 앞서 말씀드린 GCN을 사용합니다. Detection때와 마찬가지로 주로 사용하는 층(layers)들을 생성하는 함수를 `models.layers` 모듈에서 정의하고, `models.nn` 모듈에서 일반적인 Segmentation용 컨볼루션 신경망 모델을 정의하고 GCN 클래스가 상속받는 형식으로 구현하였습니다. 

### models.layers 모듈

`models.layers` 모듈은 classification 문제때와 다르게, `tf.layers` 모듈을 사용하여 간편하게 convolution layer를 재정의했으며, 이외에 GCN을 구현할 때 필요한 layer인 boundary refine layer와, global convolutional layer 그리고 upscale layer를 추가하였습니다.

```python
def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias)

def up_scale(x, scale=2):
    """feature map의 크기를 bilinear upsampling을 통해 2배 키움"""
    size = (tf.shape(x)[1]*scale, tf.shape(x)[2]*scale)
    x = tf.image.resize_bilinear(x, size)
    return tf.cast(x, x.dtype)

def boundary_refine_module(x, filters):
    """
    see: Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
    """
    c1 = conv_layer(x, filters, (3, 3), (1, 1))
    r = tf.nn.relu(c1)
    c2 = conv_layer(x, filters, (3, 3), (1, 1))
    return x + c2

def global_conv_module(x, filters, kernel_size):
    """
    see: Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
    """
    kl = kernel_size[0]
    kr = kernel_size[1]
    l1 = conv_layer(x, filters, (kl, 1), (1, 1))
    r1 = conv_layer(x, filters, (1, kr), (1, 1))
    l2 = conv_layer(l1, filters, (1, kr), (1, 1))
    r2 = conv_layer(r1, filters, (kl, 1), (1, 1))
    return l2 + r2   
```


### models.nn 모듈

`models.nn` 모듈은 마찬가지로 신경망을 표현하는 클래스를 가지고 있습니다.

#### SegNet 클래스

```python
class SegNet(metaclass=ABCMeta):
    """Segmentation을 위한 컨볼루션 신경망 모델의 베이스 클래스."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        모델 생성자.
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, 총 클래스 개수
        """
        if input_shape is None:
            input_shape = [None, None, 3]
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(
            tf.int32, [None] + input_shape[:2] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.pred = self.d['pred']
        self.logits = self.d['logits']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        모델 생성.
        해당 함수를 추후 구현해야 함.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        해당 함수를 추후 구현해야 함.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        주어진 데이터셋에 대한 예측을 수행함.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, 예측 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 예측을 위한 추가 인자.
                -batch_size: int, 각 반복 회차에서의 미니배치 크기.
        :return _y_pred: np.ndarray, shape: (N, H, W, num_classes) 
        """

        batch_size = kwargs.pop('batch_size', 64)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps + 1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(
                _batch_size, shuffle=False)
            # Compute predictions
            # (N, H, W, num_classes)
            y_pred = sess.run(self.pred, feed_dict={
                              self.X: X, self.is_train: False})
            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)

        return _y_pred
```

`SegNet` 클래스는 , 기본 추상 베이스 클래스로, 확장성을 위해 전반적인 Segmentation Network를 포괄하도록 구현하였습니다. `_build_model` 과 `_build_loss` 함수는 `SegNet`의 자식 클래스에서 구현하도록 하였고, `predict` 함수는 모델의 예측 결과를 반환합니다. 보통 Segmentation의 경우 Detection과 다르게 architecture를 구축하는 부분을 제외하고 loss 등은 크게 변하지 않습니다. 추상 클래스를 잘 정의해두면 지금 구현하는 GCN뿐만 아니라 다양한 Segmentation architecuture를 큰 시간 소비없이 손쉽게 구현할 수 있을 것이라 생각합니다!

{% include image.html name=page.name file="overview_gcn.png" description="GCN Architecture" class="full-image" %}

#### GCN 클래스

```python
class GCN(SegNet):
    """
    GCN class
    see: Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
    """
    def _build_model(self, **kwargs):
        """
        모델 생성
        :param kwargs: dict, GCN 생성을 위한 추가 인자.
                -pretrain: bool, pretrain 모델을 쓸지 말지 결정.
                -frontend: string, 불러올 pretrained model 이름.
        :return d: dict, 각 층에서의 출력값들을 포함함
        """
        d = dict()
        num_classes = self.num_classes
        pretrain = kwargs.pop('pretrain', True)
        frontend = kwargs.pop('frontend', 'resnet_v2_50')
        
        if pretrain:
            frontend_dir = os.path.join(
                'pretrained_models', '{}.ckpt'.format(frontend))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_50(
                    self.X, is_training=self.is_train)
                d['init_fn'] = slim.assign_from_checkpoint_fn(model_path=frontend_dir,
                                                              var_list=slim.get_model_variables(frontend))
                resnet_dict = [
                '/block1/unit_2/bottleneck_v2',  # conv1
                '/block2/unit_3/bottleneck_v2',  # conv2
                '/block3/unit_5/bottleneck_v2',  # conv3
                '/block4/unit_3/bottleneck_v2'   # conv4
                ]
                convs = [end_points[frontend + x] for x in resnet_dict]
        else:
            # TODO build convNet
            raise NotImplementedError("Build own convNet!")
        if self.X.shape[1].value is None:
            # input size should be bigger than (512, 512)
            g_kernel_size = (15, 15)
        else:
            g_kernel_size = (self.X.shape[1].value//32-1, self.X.shape[2].value//32-1)

        with tf.variable_scope('layer5'):
            d['gcm1'] = global_conv_module(convs[3], num_classes, g_kernel_size)
            d['brm1_1'] = boundary_refine_module(d['gcm1'], num_classes)
            d['up16'] = up_scale(d['brm1_1'], 2)

        with tf.variable_scope('layer4'):
            d['gcm2'] = global_conv_module(convs[2], num_classes, g_kernel_size)
            d['brm2_1'] = boundary_refine_module(d['gcm2'], num_classes)
            d['sum16'] = d['up16'] + d['brm2_1']
            d['brm2_2'] = boundary_refine_module(d['sum16'], num_classes)
            d['up8'] = up_scale(d['brm2_2'], 2)

        with tf.variable_scope('layer3'):
            d['gcm3'] = global_conv_module(convs[1], num_classes, g_kernel_size)
            d['brm3_1'] = boundary_refine_module(d['gcm3'], num_classes)
            d['sum8'] = d['up8'] + d['brm3_1']
            d['brm3_2'] = boundary_refine_module(d['sum8'], num_classes)
            d['up4'] = up_scale(d['brm3_2'], 2)

        with tf.variable_scope('layer2'):
            d['gcm4'] = global_conv_module(convs[0], num_classes, g_kernel_size)
            d['brm4_1'] = boundary_refine_module(d['gcm4'], num_classes)
            d['sum4'] = d['up4'] + d['brm4_1']
            d['brm4_2'] = boundary_refine_module(d['sum4'], num_classes)
            d['up2'] = up_scale(d['brm4_2'], 2)

        with tf.variable_scope('layer1'):
            d['brm4_3'] = boundary_refine_module(d['up2'], num_classes)
            d['up1'] = up_scale(d['brm4_3'], 2)
            d['brm4_4'] = boundary_refine_module(d['up1'], num_classes)

        with tf.variable_scope('output_layer'):
            d['logits'] = conv_layer(d['brm4_4'], num_classes, (1, 1), (1, 1))
            d['pred'] = tf.nn.softmax(d['logits'], axis=-1)

        return d

    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성
        :return tf.Tensor.
        """
        # pixel-wise cross entropy loss를 계산하고 unknown 영역을 무시
        ignore = tf.cast(tf.greater_equal(self.y[...,0], 0), dtype=tf.float32)
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.y, logits=self.logits, dim=-1)
        loss = tf.multiply(softmax_loss, ignore)
        return tf.reduce_mean(loss)

```

`GCN` 모델은 pretrained model에서 encoding된 압축된 feature map을 원래 이미지 크기로 decoding하면서 segmentation map을 생성하고 각 pixel마다 클래스를 분류하여 이미지를 분할합니다. GCN(Global convolutional Network)은 이름 그대로 pixel을 예측할 때 global하게 보기 위하여 kernel 사이즈를 이미지 전체만큼 가져가며 ($$512 \times 512$$의 경우 $$15 \times 15$$) 이를 정제하는 방식으로 decoding 합니다. 첨부한 모식도를 보시면 보다 더 쉽게 이해하실 수 있을 것 같습니다.

손실 함수의 경우 classification때와 마찬가지로 cross entropy loss를 사용합니다. Classification에서 image 단위로 cross entropy를 계산했다면, Segmentation에서는 pixel 단위로 cross entropy를 계산하여 사용합니다. 사용하는 데이터셋의 경우 Unknown region이 있기 때문에, Unknown pixel은 loss 계산에서 제외하는 방식으로 손실 함수를 정의하였습니다.

추가로, Background에 비해 Foreground 영역이 작아 imbalance한 경우 그리고 binary segmentation인 경우, 1-Dice coefficient 혹은 1-IoU를 손실함수로 사용할 수도 있습니다. 풀고자하는 문제가 위와 같다면 한번 손실함수를 재정의해서 학습에 사용해보시는 것도 추천 드립니다.  

** 제가 구현한 GCN의 경우 resnetv2-50(slim ver.) pretrained model을 사용합니다. 제공된 구현체를 수정없이 돌려보시려면 <a href="https://github.com/tensorflow/models/tree/master/research/slim" target="_blank">여기</a>에 들어가서 pretrained model을 받은 후, *pretrained_models* 폴더에 체크포인트(.ckpt)파일을 넣어두시면 됩니다! 추가로, 자기만의 Convolutional network를 구현하거나 다른 pretrained model도 사용해보시길 권장드립니다. 

## (4) 러닝 알고리즘: SGD+Momentum

러닝 알고리즘은 Classification 문제 그리고 Detection 문제때와 크게 다르지 않습니다. **모멘텀(momentum)**을 적용한 **확률적 경사 하강법(stochastic gradient descent; 이하 SGD)**을 채택하였으며, 베이스 클래스를 먼저 정의한 뒤, 이를 모멘텀 SGD에 기반한 optimizer 클래스가 상속받는 형태로 구현하였습니다. Pretrained model weights를 불러오는 부분을 제외하고 Detection 포스팅 때와 동일하니 설명은 생략하도록 하겠습니다.

### learning.optimizers 모듈

#### Optimizer 클래스

```python
class Optimizer(metaclass=ABCMeta):
    """경사 하강 러닝 알고리즘 기반 optimizer의 베이스 클래스"""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer 생성자.
        :param model: Net, 학습할 모델.
        :param train_set: DataSet, 학습에 사용할 학습 데이터셋.
        :param evaluator: Evaluator, 학습 수행 과정에서 성능 평가에 사용할 evaluator.
        :param val_set: Datset, 검증 데이터셋, 주어지지 않은 경우 None으로 남겨둘 수 있음.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
                - batch_size: int, 각 반복 회차에서의 미니배치 크기.
                - num_epochs: int, 총 epoch 수.
                - init_learning_rate: float, 학습률 초기값.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # 학습 관련 하이퍼파라미터
        self.batch_size = kwargs.pop('batch_size', 32)
        self.num_epochs = kwargs.pop('num_epochs', 300)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.001)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """일부 변수를 재설정."""
        self.curr_epoch = 1
        # number of bad epochs, where the model is updated without improvement.
        self.num_bad_epochs = 0
        # initialize best score with the worst one
        self.best_score = self.evaluator.worst_score
        self.curr_learning_rate = self.init_learning_rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.Optimizer.minimize Op.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        고유의 학습률 스케줄링 방법에 따라, (필요한 경우) 매 epoch마다 현 학습률 값을 업데이트함.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        경사 하강 업데이트를 1회 수행하며, 관련된 값을 반환함.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        :param sess, tf.Session.
        :return loss: float, 1회 반복 회차 결과 손실 함수값.
                y_true: np.ndarray, 학습 데이터셋의 실제 레이블.
                y_pred: np.ndarray, 모델이 반환한 예측 레이블.
        """

        # 미니배치 하나를 추출함
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True)
        # 손실 함수값을 계산하고, 모델 업데이트를 수행.
        _, loss, y_pred = \
            sess.run([self.optimize, self.model.loss, self.model.pred_y],
                     feed_dict={self.model.X: X, self.model.y: y_true, self.model.is_train: True, self.learning_rate_placeholder: self.curr_learning_rate})
        return loss, y_true, y_pred, X

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Optimizer를 실행하고, 모델을 학습함.
        :param sess: tf.Session.
        :param save_dir: str, 학습된 모델의 파라미터들을 저장할 디렉터리 경로.
        :param details: bool, 학습 결과 관련 구체적인 정보를, 학습 종료 후 반환할지 여부.
        :param verbose: bool, 학습 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
        :return train_results: dict, 구체적인 학습 결과를 담은 dict
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())  # 전체 파라미터들을 초기화함
        
        # pretrained weight를 가져옴
        if 'init_fn' in self.model.d.keys():
            print('Load pretrained weights...')
            self.model.d['init_fn'](sess)
            
        train_results = dict()
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch
        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # 학습 루프를 실행함
        for i in range(num_steps):
            # 미니배치 하나로부터 경사 하강 업데이트를 1회 수행함
            step_loss, step_y_true, step_y_pred = self._step(sess)
            step_losses.append(step_loss)
            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i) % num_steps_per_epoch == 0:
                # 학습 데이터셋으로부터 추출한 현재의 미니배치에 대하여 모델의 예측 성능을 평가함
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # 검증 데이터셋이 주어진 경우, 이를 사용하여 모델 성능을 평가함
                if self.val_set is not None:
                    # 검증 데이터셋을 사용하여 모델 성능을 평가함
                    eval_y_pred = self.model.predict(
                        sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(
                        self.val_set.labels, eval_y_pred, **kwargs)
                    eval_scores.append(eval_score)

                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # 중간 결과를 플롯팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)

                    curr_score = eval_score
                # 그렇지 않은 경우, 단순히 미니배치에 대한 결과를 사용하여 모델 성능을 평가함
                else:
                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |lr: {:.6f}'\
                            .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # 중간 결과를 플롯팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)

                    curr_score = step_score

                # 현재의 성능 점수의 현재까지의 최고 성능 점수를 비교하고,
                # 최고 성능 점수가 갱신된 경우 해당 성능을 발휘한 모델의 파라미터들을 저장함
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(
                time.time() - start_time))
            print('Best {} score: {}'.format(
                'evaluation' if eval else 'training', self.best_score))
        print('Done.')


        if details:
            # 학습 결과를 dict에 저장함
            train_results['step_losses'] = step_losses
            train_results['step_scores'] = step_scores
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores

            return train_results
```

#### MomentumOptimizer 클래스

```python
class MomentumOptimizer(Optimizer):
    """모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스."""

    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op.
       :param kwargs: dict, optimizer의 추가 인자.
                -momentum: float, 모멘텀 계수.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_vars = tf.trainable_variables()
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.AdamOptimizer(self.learning_rate_placeholder, momentum).minimize(
                self.model.loss, var_list=update_vars)
        return train_op

    def _update_learning_rate(self, **kwargs):
        """
        성능 평가 점수 상에 개선이 없을 때, 현 학습률 값을 업데이트함.
        :param kwargs: dict, 학습률 스케줄링을 위한 추가 인자.
            - learning_rate_patience: int, 성능 향상이 연속적으로 이루어지지 않은 epochs 수가 
                                      해당 값을 초과할 경우, 학습률 값을 감소시킴.
            - learning_rate_decay: float, 학습률 업데이트 비율.
            - eps: float, 업데이트된 학습률 값과 기존 학습률 값 간의 차이가 해당 값보다 작을 경우,
                          학습률 업데이트를 취소함.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # 새 학습률 값과 기존 학습률 값 간의 차이가 eps보다 큰 경우에 한해서만 업데이트를 수행함
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
```


## 학습 수행 및 테스트 결과

`train.py` 스크립트에서 실제 학습을 수행하는 과정을 구현하며, `test.py` 스크립트에서 테스트 데이터셋에 대해 학습이 완료된 모델을 테스트하여 성능 수치를 보여주고 실제로 Segmentation 마스크도 그려줍니다. 혹, 레이블이 없는 데이터셋에 대해서 그려보고 싶은 분들을 위해 `draw.py` 스크립트도 추가 구현하였으니 저장소에서 참고하시길 바랍니다.

### train.py 스크립트

```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('data/catdog/') # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 앵커 로드
anchors = dataset.load_json(os.path.join(trainval_dir, 'anchors.json'))

# 학습에 사용될 이미지 사이즈 및 클래스 개수를 정함
IM_SIZE = (512, 512)
NUM_CLASSES = 3

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_data(trainval_dir, IM_SIZE)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.1) # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정"""
hp_d = dict()

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 8
hp_d['num_epochs'] = 300
hp_d['init_learning_rate'] = 1e-3
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
hp_d['pretrain'] = True

""" 3. Graph 생성, session 초기화 및 학습 시작 """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, **hp_d)

evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
```

`train.py` 스크립트에서는 마찬가지로 3단계로 진행됩니다.

1. 원본 학습 데이터셋을 메모리에 로드하고, 이를 학습 데이터셋(90%)과 검증 데이터셋(10%)으로 나눠 객체 생성. 
2. 학습 관련 하이퍼파라미터 설정.
3. `ConvNet` 객체, `Evaluator` 객체 및 `Optimizer` 객체를 생성하고, TensorFlow Graph와 Session을 초기화한 뒤, `Optimizer.train` 함수를 호출하여 모델 학습을 수행함

* 원본 데이터셋 저장 경로, 하이퍼파라미터 등 `FIXME`로 표시된 부분은 여러분의 상황에 맞게 수정하시면 됩니다.

### test.py 스크립트

```python
""" 1. 원본 데이터셋을 메모리에 로드함 """
root_dir = os.path.join('data/catdog/')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# 학습에 사용될 이미지 사이즈 및 클래스 개수를 정함
IM_SIZE = (512, 512)
NUM_CLASSES = 3

# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_data(test_dir, IM_SIZE)
test_set = dataset.DataSet(X_test, y_test)

""" 2. 테스트를 위한 하이퍼파라미터 설정 """
hp_d = dict()

# FIXME
hp_d['batch_size'] = 8

""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, './model.ckpt')    # 학습한 weight 로드
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test performance: {}'.format(test_score))

""" 4. 이미지 마스킹 """
draw_dir = os.path.join(test_dir, 'draws') # FIXME
if not os.path.isdir(draw_dir):
    os.mkdir(draw_dir)
im_dir = os.path.join(test_dir, 'images') # FIXME
im_paths = []
im_paths.extend(glob.glob(os.path.join(im_dir, '*.jpg')))
test_outputs = draw_pixel(test_y_pred)
test_results = test_outputs + test_set.images
for img, im_path in zip(test_results, im_paths):
    name = im_path.split('/')[-1]
    draw_path =os.path.join(draw_dir, name)
    cv2.imwrite(draw_path, img)

```
`test.py` 스크립트도 비슷하게 4단계 과정을 거쳐 성능을 측정하고 이미지에 예측된 마스크를 덧입혀 시각화하여 저장하였습니다.

## 학습 결과 분석

### 학습 곡선

Classification 문제때와 마찬가지로 학습 수행 과정동안 학습 곡선을 그려보았습니다. 

{% include image.html name=page.name file="plot.png" description="학습 곡선 플롯팅 결과<br><small>(파란색: 학습 데이터셋 정확도, 빨간색: 검증 데이터셋 정확도)</small>" class="large-image" %}

학습이 진행됨에 따라, Loss는 점차 떨어지고 성능 지표는 pretrained model을 사용해서 초반부터 높긴 하지만, 점차 올라가는 전형적인 학습 곡선 양상을 확인할 수 있었습니다. 검증 셋에 대해 가장 성능 지표가 높은 모델을 선택하고 이를 테스트셋에 대해 활용하여 테스트 성능을 다시 확인하였습니다.

### 테스트 결과

테스트 결과 측정된 Pixel accuracy 값은 **0.9556**로 꽤 높은 값을 가졌습니다. 하지만 Pixel accuracy는 사용하는 데이터셋의 Foreground가 큰 경우에도 불구하고, 배경으로 모두 예측한 경우에도 꽤 높은 값을 가질 수 있기 때문에 값만 보고 추측하기는 어려울 때가 많습니다. 나중에 추가로 mIoU(mean Intersection over Union)등을 성능 지표로 삼고 다시 체크해보시길 추천드립니다. 돌아와서, 성능 값만 가지고 모델이 정말 잘 예측하는 지 신뢰하기 힘들기 때문에 실제 테스트 이미지에서 정말 객체 분할 및 분류를 잘했는지 확인하기 위하여 실제 이미지에 예측한 마스크를 시각화 해보았습니다.

{% include image.html name=page.name file="correct.png" description="정확하게 예측한 결과 예시" class="full-image" %}

{% include image.html name=page.name file="incorrect.png" description="미흡하게 예측한 결과 예시" class="full-image" %}

대부분의 이미지에서 개 혹은 고양이 객체를 픽셀단위로 꽤 정확히 예측하였습니다. 하지만, 간혹 고양이가 카펫에 누워있거나, 배경이 헷갈리는 경우 배경까지 객체로 인식하는 경우 혹은 개를 고양이로, 고양이를 개로 인식하는 경우도 간혹가다 있었습니다. 이를 해결하기 위해 추가적인 Augmentation 기법이나 데이터를 늘리는 등 Fine-tuning을 거친다면 보다 더 좋은 결과를 낼 수 있다고 생각합니다!

## 결론

본 포스팅에서는 이미지 인식 분야에서 중요하게 다뤄지는 Segmentation 문제를 응용할 수 있는 **개vs고양이 분할 문제** 사례를 소개하고 이를 GCN 모델과 TensorFlow를 이용한 딥러닝 알고리즘으로 해결하는 과정을 간단하게 안내해드렸습니다. 실제 Segmentation을 해야할 상황보다 쉬운 Task이기 때문에 엄밀하게 풀어냈다라고 할 수는 없지만, 실제 Segmentation 문제를 공부하고 처음 구현해보시는 분들에게는 약간이나마 도움이 됐을 것이라 생각합니다. 이론적인 부분이나 구현에서 이해안되는 부분있으시면 언제든 댓글 남겨주세요.

** 이로써 이미지 인식 분야에서 다뤄지는 세가지 문제(Classification, Detection, Segmentation)에 대해 간단한 데이터셋을 가지고 데이터 로드부터 구현 및 성능 체크과정까지 모두 다뤄보았습니다. 기획한 이미지 인식 분야 문제 해결과정에 대한 글은 여기서 마무리하였지만, 추가로 딥러닝에 관하여 원하시는 주제가 있다면 언제든 댓글 혹은 연락주시면 감사하겠습니다! 

## References

- GCN 논문
  - <a href="https://arxiv.org/abs/1703.02719" target="_blank">
Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network</a>
- The Oxford-IIIT Pet Dataset
  - <a href="http://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank">The Oxford-IIIT Pet Dataset</a>
