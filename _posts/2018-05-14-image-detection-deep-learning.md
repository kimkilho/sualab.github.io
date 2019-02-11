---
layout: post
title: "이미지 Detection 문제와 딥러닝: YOLOv2로 얼굴인식하기"
date: 2018-05-14 09:00:00 +0900
author: jongsoo_keum
categories: [Practice]
tags: [detection, yolov2, tensorflow]
comments: true
name: image-detection-deep-learning
---

안녕하세요, 오랜만에 포스팅합니다. 이전 글인 Classification 문제에 이어 딥러닝을 적용하여 Detection 문제를 해결한 사례를 앞선 포스팅과 마찬가지로 **Tensorflow** 구현 코드와 함께 소개해드리겠습니다. 이미 많은 포스팅에서 **Detection**에 대한 설명과 그 중 한 방법론인 **YOLO(You Only Look Once)**의 개념 및 특징에 대해 훌륭한 설명이 많기 때문에 같은 설명을 반복하기 보다 개념적인 설명은 조금 뒤로하고 실제 구현 코드와 이를 뒷받침하는 설명을 중심으로 진행하도록 하겠습니다. 혹 설명이 부족하여 이해가 안되는 부분들 댓글달아주시면 답변드리겠습니다.

* **다음과 같은 사항을 알고계시면 더 이해하기 쉽습니다.**
  - 딥러닝에 대한 전반적인 이해
  - Python 언어 및 TensorFlow 프레임워크에 대한 이해
* 이번 글에서 구현한 YOLOv2의 경우, 논문에서 명시된 것처럼 ImageNet challenge 데이터셋으로 학습한 pre-trained model을 사용하지 않고, 학습방법을 단순화하여 성능이 논문과 상이할 가능성이 있습니다.
* 이번 글에서는 과거 Classification 구현체와 마찬가지로 데이터셋(data set), 성능 평가(performance evaluation), 러닝 모델(learning model), 러닝 알고리즘(leaning algorithm) 4가지 요소를 나눠 구현하였으며, 중복을 피하기 위해 다르게 구현한 부분 위주로 설명합니다. 
  - 전체 구현체 코드는 <a href="https://github.com/sualab/object-detection-yolov2" target="_blank">수아랩의 GitHub 저장소</a>에서 자유롭게 확인하실 수 있습니다.
  - 본 글에서 사용한 얼굴인식 데이터셋은 이곳에서 다운로드 받으실 수 있으며, 저장소에 있는 스크립트(ellipsis\_to\_rectangle.py)를 참고하여 일반적인 Detection 문제에서 사용되는 Rectangle 형태의 annotation으로 변환하실 수 있습니다. 
  - 제가 변환한 데이터셋은 <a href="https://drive.google.com/file/d/1qV4YSzvvTQ7rSi3iS2swkbA56QO-bbs8/view?usp=sharing" target="_blank">여기</a>서 받을 수 있습니다. 

## 서론

<a href="{{ site.url }}/computer-vision/2017/11/29/image-recognition-overview-1.html" target="_blank">이미지 인식 문제의 개요: PASCAL VOC Challenge를 중심으로</a>에서 언급한 바와 같이, **PASCAL VOC challenge**에서 중요하게 다루는 3가지 이미지 인식 문제 중 Classification에 이어서 **Detection**기술로 해결할 수 있는 간단한 사례를 소개하고, 이를 딥러닝 기술 중 많은 분들이 접해본 **YOLO**계열 기술을 통해 해결하는 과정을 설명드리겠습니다. 

앞서 말씀드린 것과 같이 이번 포스팅은 개념적인 설명부터 시작하면 지나치게 글이 길어지고 집중도가 떨어질 것 같아 (1) 딥러닝 및 머신러닝에 대한 어느정도의 이해와, (2) Python 언어 및 TensorFlow 프레임워크에 대한 이해, 그리고 (3) YOLO계열 Detection 알고리즘에 대한 전반적인 이해를 알고 있다는 전제하에 글을 쓰겠습니다. 글을 읽다가 이해가 안되는 부분이 있다면 다른곳에 좋은 글이 많이 있으니 숙지하신 후 읽어보시길 권장드립니다.

본 포스팅에서 다룰 Detection문제로 이미 앞서 소개했고 많은 분들에게 알려진 PASCAL VOC Challenge 혹은 MS COCO Challenge 데이터셋으로 소개해드리려 했으나, 이들 데이터셋은 상당히 방대하고 지나치게 학습시간이 오래걸려 적합하지 않다고 판단하였습니다. 이에 실용적이며 간단하고 쉽게 확장이 가능한 **얼굴 인식** 데이터셋을 이용하여 문제를 단순화하여 진행하려고 합니다.

**얼굴 인식** 문제를 해결하기 위한 딥러닝 알고리즘으로는 **YOLO**계열의 두번째 버전인 **YOLOv2**(**YOLO9000**)를 채택하였습니다. YOLO 모델의 경우 조금 시간이 지난 모델이긴 하지만, 속도면에서 요즘 모델과 비교해도 여전히 빠르며 성능도 여전히 준수한 편이며 직관적이기 때문에 이해하고 적용하는 데 **R-CNN**계열 혹은 **SSD(Single Shot Detector)**보다 나을 것이라 판단하였습니다.

YOLOv2 구현체는 앞선 <a href="{{ site.url }}/machine-learning/computer-vision/2018/01/17/image-classification-deep-learning.html" target="_blank">Classification 문제</a>와 같이 데이터셋(data set), 성능 평가(performance evaluation), 러닝 모델(learning model), 러닝 알고리즘(leaning algorithm) 4가지 요소를 중심으로 작성하였으며  이전 포스팅과 겹치지 않은 부분 위주로 소개해드리겠습니다.

## (1) 데이터셋: 얼굴인식 (Face Detection)

얼굴인식 문제를 위해 사용한 데이터셋은 <a href="http://vis-www.cs.umass.edu/fddb/" target="_blank">FDDB: Face Detection Data Set and Benchmark(FDDB)</a>에서 가져왔습니다. 데이터셋의 원본 annotation의 경우 다음 <a href="http://vis-www.cs.umass.edu/fddb/samples/" target="_blank">샘플 예제</a>와 같이 타원형으로 annotation이 되어있기 때문에 일반적인 Detection annotation과 상이합니다. 이를 YOLO 모델에 쉽게 적용하기 위해서는 Rectangle형태로 변환해주는 것이 좋기 때문에 먼저 이 변환작업을 해주는 것을 추천드립니다. 

Annotation 표현 방법은 무수히 많지만, 이번 포스팅 예제에서는 다음과 같은 형식으로 진행하도록 하겠습니다. 
* 이미지마다 매칭되는 **anno 확장자** 파일 생성합니다. ex) sample1.png <--> sample1.anno
* anno 파일은 **json 형식**으로, 이름은 클래스 종류를 나타내고, 값은 array 형식으로 [x\_min, y\_min, x\_max, y\_max] 좌표 값을 가지고 있습니다. ex) 한 이미지에 얼굴 객체가 두개 있다면, { "face" : [ [20, 40, 78, 100], [150, 40, 170, 70] ] } 와 같은 형식으로 저장됩니다.
* 폴더에 images 폴더, annotations 폴더, anchors.json 파일(추후 설명), classes.json 파일을 두고 images 폴더에 이미지를, annotations 폴더에 위의 형식의 파일을 넣어둡니다.
* 자세한 사항은 ellipsis\_to\_rectangle.py 스크립트를 참고하시면 도움이 될 것 같습니다.
* YOLOv2에서 사용되는 k-means 기반 앵커(anchor)는 앵커 계산에 관한 스크립트(calculate_anchor_boxes.py)를 참고하시면 이해에 많은 도움이 됩니다.

{% include image.html name=page.name file="example.png" description="얼굴 인식 데이터셋 예시 (annotation 변환 후)" class="full-image" %}

데이터셋은 총 2865장으로 이중 임의로 10%를 골라 약 230여장을 테스트셋으로 사용하였습니다. 클래스는 "얼굴" 한 가지로 객체가 있을 곳을 예측하고 얼굴인지 아닌지 판단하는 절차로 진행됩니다. 

### datasets.data 모듈

`datasets.data` 모듈은 데이터셋에 관련된 함수와 클래스를 가지고 있습니다. Classification 문제때와 마찬가지로, 이 모듈은 데이터셋을 메모리에 로드하고 학습 및 예측평가 과정에서 미니배치(minibatch) 단위로 제공해주는 역할을 합니다.

#### read\_data 및 get\_best\_anchor 함수

```python
 def read_data(data_dir, image_size, pixels_per_grid=32, no_label=False):
    """
    YOLO 디텍터를 위해 data를 로드하고, 전처리 수행
    :param data_dir: str, annotation, image 등 데이터가 저장된 경로
    :image_size: tuple, 리사이징하기 위해 지정된 이미지 사이즈
    :pixels_per_gird: int, 한 그리드당 실제 사이즈
    :no_label: bool, 레이블을 로드할 지 여부
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes).
    """
    im_dir = os.path.join(data_dir, 'images')
    class_map_path = os.path.join(data_dir, 'classes.json')
    anchors_path = os.path.join(data_dir, 'anchors.json')
    class_map = load_json(class_map_path)
    anchors = load_json(anchors_path)
    num_classes = len(class_map)
    grid_h, grid_w = [image_size[i] // pixels_per_grid for i in range(2)]
    im_paths = []
    for ext in IM_EXTENSIONS:
        im_paths.extend(glob.glob(os.path.join(im_dir, '*.{}'.format(ext))))
    anno_dir = os.path.join(data_dir, 'annotations')
    images = []
    labels = []

    for im_path in im_paths:
        # 이미지를 로드하고 지정된 사이즈로 변환
        im = imread(im_path)
        im = np.array(im, dtype=np.float32)
        im_origina_sizes = im.shape[:2]
        im = resize(im, (image_size[1], image_size[0]))
        if len(im.shape) == 2:
            im = np.expand_dims(im, 2)
            im = np.concatenate([im, im, im], -1)
        images.append(im)

        if no_label:
            labels.append(0)
            continue
        # 바운딩 박스를 로드하고 YOLO 모델에 맞게 변환
        name = os.path.splitext(os.path.basename(im_path))[0]
        anno_path = os.path.join(anno_dir, '{}.anno'.format(name))
        anno = load_json(anno_path)
        label = np.zeros((grid_h, grid_w, len(anchors), 5 + num_classes))
        for c_idx, c_name in class_map.items():
            if c_name not in anno:
                continue
            for x_min, y_min, x_max, y_max in anno[c_name]:
                oh, ow = im_origina_sizes
                # 좌표를 0~1사이로 노말라이즈하고 벗어나지 않게 클립
                x_min, y_min, x_max, y_max = x_min / ow, y_min / oh, x_max / ow, y_max / oh
                x_min, y_min, x_max, y_max = np.clip([x_min, y_min, x_max, y_max], 0, 1)
                # 값을 최적에 앵커에 지정
                anchor_boxes = np.array(anchors) / np.array([ow, oh])
                best_anchor = get_best_anchor(
                    anchor_boxes, [x_max - x_min, y_max - y_min])
                cx = int(np.floor(0.5 * (x_min + x_max) * grid_w))
                cy = int(np.floor(0.5 * (y_min + y_max) * grid_h))
                label[cy, cx, best_anchor, 0:4] = [x_min, y_min, x_max, y_max]
                label[cy, cx, best_anchor, 4] = 1.0
                label[cy, cx, best_anchor, 5 + int(c_idx)] = 1.0
        labels.append(label)

    X_set = np.array(images, dtype=np.float32)
    y_set = np.array(labels, dtype=np.float32)

    return X_set, y_set

def get_best_anchor(anchors, box_wh):
    """
    가장 높은 IOU를 가지는 anchor를 선택
    """
    box_wh = np.array(box_wh)
    best_iou = 0
    best_anchor = 0
    for k, anchor in enumerate(anchors):
        intersect_wh = np.maximum(np.minimum(box_wh, anchor), 0.0)
        intersect_area = intersect_wh[0] * intersect_wh[1]
        box_area = box_wh[0] * box_wh[1]
        anchor_area = anchor[0] * anchor[1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        if iou > best_iou:
            best_iou = iou
            best_anchor = k
    return best_anchor
```

`load_data` 함수는 위의 형식으로 저장된 데이터셋을 불러와 각 이미지를 원하는 크기로 변환한뒤, ndarray 형태로 반환합니다. 마찬가지로 바운딩 박스가 저장된 annotation 파일을 불러와 get\_best\_anchor 함수를 이용하여 최적의 anchor에 노말라이즈(normalize)된 바운딩 박스 좌표를 지정하여 ndarray 형태로 반환합니다. 만약 이미지 사이즈가 (416, 416)이고, 앵커가 5개라면 Feature extractor인 DarkNet이 이미지 사이즈를 32배 줄여주기 때문에 그리드맵은 (13, 13)이 되며, 검출할 객체의 센터가 위치한 그리드 중 가장 IOU가 높은 앵커가 물체를 검출할 책임을 갖게 됩니다. 즉, y_set의 형태는 (N, 13, 13, 5, 좌표(4) + confidence(1) + class 개수)로 표현됩니다. 

#### DataSet 클래스

```python
class DataSet(object):

    def __init__(self, images, labels=None):
        """
        새로운 DataSet 객체를 생성함.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes).
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
        :param batch_size: int, 미니배치 크기
        :param shuffle: bool, 추출 이전에, 데이터셋 이미지를 섞을지 여부
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
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
        :param batch_size: int,  미니배치 크기
        :param shuffle: bool, 추출 이전에, 데이터셋 이미지를 섞을지 여부
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
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
Classification 문제때와 마찬가지로 DataSet 클래스를 이용하여 메모리에 로드된 X_set과 y_set을 미니배치(minibatch) 단위로 반환해 줍니다.

### (2) 성능 평가: 재현율(Recall)

모델의 얼굴인식 성능 평가를 위해 **재현율(Recall)**을 사용합니다. Detection에서는 mAP(mean average precision)가 가장 빈번하게 사용되는 성능 척도이나, 클래스가 한가지밖에 없어 평균의 의미가 없고 물체로 인식한 대상이 높은 확률로 분류까지 하는 지 확인할 수 있는 재현율이 더 직관적이라 판단하여 재현율을 성능 평가 척도로 사용하였습니다. 요약하면, 테스트를 위해 주어진 전체 오브젝트 수 대비, 실제 ground truth 바운딩 박스와 예측한 바운딩 박스의 IOU(Intersection of Union)가 특정 임계값 이상일 때 올바르게 분류한 오브젝트 수로 재현율을 정의합니다.

\begin{equation}
\text{Recall} = \frac{\text{위치를 찾고 올바르게 분류한 오브젝트 수}} {\text{전체 오브젝트 수}}
\end{equation} 

### learning.evaluators 모듈

Classification 문제와 마찬가지로, 성능 평가를 위한 'evaluator' 클래스를 담고 있습니다.

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

#### RecallEvaluator 클래스

```python
class RecallEvaluator(Evaluator):
    """ 재현율(Recall)을 성능 평가 청도로 사용하는 evaluator 클래스"""

    @property
    def worst_score(self):
        """최저 성능 점수"""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지 낮아야 우수한지 여부"""
        return 'max'

    def score(self, y_true, y_pred, **kwargs):
        """
        주어진 바운딩 박스에 대한 Recall 성능 평가 점수
        :param kwargs: dict, 추가 인자.
            - nms_flag: bool, True면 nms 수행
        """
        nms_flag = kwargs.pop('nms_flag', True)
        if nms_flag:
            bboxes = predict_nms_boxes(y_pred)
        else:
            bboxes = convert_boxes(y_pred)
        gt_bboxes = convert_boxes(y_true)
        score = cal_recall(gt_bboxes, bboxes)
        return score

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


재현율을 성능 평가 척도로 사용하기 위해 상속받아 RecallEvaluator 클래스를 구현하였습니다. `score` 함수에서 재현율을 계산하기 위해, learning.utils 모듈에 cal_recall 함수를 구현해두었습니다. 재현율은 높을 수록 좋기에 `mode` 함수는 max로 `is_better` 함수는 특정 임계값을 넘었을 때 최고 성능이 바뀌도록 구현하였습니다.


## 러닝 모델: YOLOv2

러닝 모델로는 앞서 말씀드린 YOLOv2를 사용합니다. 이전과 마찬가지로 주로 사용하는 층(layers)들을 생성하는 함수를 models.layers 모듈에서 정의하고, models.nn 모듈에서 일반적인 Detection용 컨볼루션 신경망 모델을 정의하고 YOLOv2 클래스가 상속받는 형식으로 구현하였습니다. 

### models.layers 모듈

`models.layers` 모듈은 classification 문제때와 거의 같으며, 추가로 batchNormalization layer를 함수로 정의하였습니다.

```python
def weight_variable(shape, stddev=0.01):
    """
    새로운 가중치 변수를 주어진 shape에 맞게 선언하고,
    Normal(0.0, stddev^2)의 정규분포로부터의 샘플링을 통해 초기화함.
    :param shape: list(int).
    :param stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    새로운 바이어스 변수를 주어진 shape에 맞게 선언하고, 
    주어진 상수값으로 추기화함.
    :param shape: list(int).
    :param value: float, 바이어스의 초기화 값.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    주어진 입력값과 필터 가중치 간의 2D 컨볼루션을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    주어진 입력값에 대해 최댓값 풀링(max pooling)을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 풀링 윈도우의 한 변의 길이.
    :param stride: int, 풀링 윈도우의 각 방향으로의 이동 간격. 
    :param padding: str, 'SAME' 또는 'VALID',
                         풀링 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', use_bias=True, **kwargs):
    """
    새로운 컨볼루션 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 필터 한 변의 길이
    :param stride: int, 필터의 각 방향으로의 이동 간격
    :param out_depth: int, 입력값에 적용할 필터의 총 개수
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘
    :param use_bias: bool, True이면, 바이어스 값을 사용, 아니면 사용하지 않음.
                          전형적으로 batchnormalization 층이 이후에 나오면 바이어스를 사용하지 않음.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함.
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    in_depth = int(x.get_shape()[-1])
    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    if use_bias:
        biases_value = kwargs.pop('biases_value', 0.1)
        biases = bias_variable([out_depth], value=biases_value)
        return conv2d(x, filters, stride, padding=padding) + biases
    else:
        return conv2d(x, filters, stride, padding=padding)

def batchNormalization(x, is_train):
    """
    새로운 batchNormalization 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), True이면 train mode, 아니면 test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.99, epsilon=0.001, center=True, scale=True)

def fc_layer(x, out_dim, **kwargs):
    """
    새로운 완전 연결 층을 추가함.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, 출력 벡터의 차원수.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함. 
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases
```
BatchNormalization 층은 보통 컨볼루션 층 이후에 나오며, 이 때 컨볼루션에서 보통 추가로 더해지는 바이어스는 필요하지 않습니다. 이에 conv\_layer 함수를 일부 수정하였습니다.

### models.nn 모듈

`models.nn` 모듈은 마찬가지로 신경망을 표현하는 클래스를 가지고 있습니다.

#### DetectNet 클래스
```python
class DetectNet(metaclass=ABCMeta):
    """Detection을 위한 컨볼루션 신경망 모델의 베이스 클래스."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        모델 생성자.
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, 총 클래스 개수
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.pred = self.d['pred']
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
        :return _y_pred: np.ndarray, shape (N, 5 + number of classes) 
        """

        batch_size = kwargs.pop('batch_size', 16)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size
        flag = int(bool(pred_size % batch_size))
        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+flag):
            if i == num_steps and flag:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False)

            # Compute predictions
            # (N, grid_h, grid_w, 5 + num_classes)
            y_pred = sess.run(self.pred_y, feed_dict={
                              self.X: X, self.is_train: False})

            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)
        return _y_pred
```

`DetectNet` 클래스는 , 기본 추상 베이스 클래스로, 확장성을 위해 전반적인 Detection Network를 포괄하도록 구현하였습니다. `_build_model` 과 `_build_loss` 함수는 `DetectNet`의 자식 클래스에서 구현하도록 하였고, `predict` 함수는 모델의 예측 결과를 반환합니다. 보통 Detection의 예측 결과는 nms(Non Maximum Suppression)을 거치지 않았을 때 총 바운딩 박스마다 좌표와 클래스 확률을 나타내는데, **YOLO**의 경우 background class의 확률이 없는 대신, 바운딩 박스의 신뢰도(confidence) 점수를 가지고 있습니다. 하지만 전반적인 모양은 거의 같기 때문에 베이스 클래스에서 구현하였습니다.

#### YOLO 클래스
```python
class YOLO(DetectNet):
    """YOLO class"""
    def __init__(self, input_shape, num_classes, anchors, **kwargs):
        # YOLO 신경망을 위한 추가 생성자
        self.grid_size = grid_size = [x // 32 for x in input_shape[:2]]
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.y = tf.placeholder(tf.float32, [None] +
                                [self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + num_classes])
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        """
        모델 생성
        :param kwargs: dict, YOLO 생성을 위한 추가 인자.
                -image_mean: np.ndarray, 평균 이미지: 이미지들의 각 일벽 채널별 평균값, shape: (C,).
        :return d: dict, 각 층에서의 출력값들을 포함함
        """
        d = dict()
        x_mean = kwargs.pop('image_mean', 0.0)

        # input
        X_input = self.X - x_mean
        is_train = self.is_train

        #conv1 - batch_norm1 - leaky_relu1 - pool1
        with tf.variable_scope('layer1'):
            d['conv1'] = conv_layer(X_input, 3, 1, 32,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm1'] = batchNormalization(d['conv1'], is_train)
            d['leaky_relu1'] = tf.nn.leaky_relu(d['batch_norm1'], alpha=0.1)
            d['pool1'] = max_pool(d['leaky_relu1'], 2, 2, padding='SAME')
        # (416, 416, 3) --> (208, 208, 32)
        print('layer1.shape', d['pool1'].get_shape().as_list())

        #conv2 - batch_norm2 - leaky_relu2 - pool2
        with tf.variable_scope('layer2'):
            d['conv2'] = conv_layer(d['pool1'], 3, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm2'] = batchNormalization(d['conv2'], is_train)
            d['leaky_relu2'] = tf.nn.leaky_relu(d['batch_norm2'], alpha=0.1)
            d['pool2'] = max_pool(d['leaky_relu2'], 2, 2, padding='SAME')
        # (208, 208, 32) --> (104, 104, 64)
        print('layer2.shape', d['pool2'].get_shape().as_list())

        #conv3 - batch_norm3 - leaky_relu3
        with tf.variable_scope('layer3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm3'] = batchNormalization(d['conv3'], is_train)
            d['leaky_relu3'] = tf.nn.leaky_relu(d['batch_norm3'], alpha=0.1)
        # (104, 104, 64) --> (104, 104, 128)
        print('layer3.shape', d['leaky_relu3'].get_shape().as_list())

        #conv4 - batch_norm4 - leaky_relu4
        with tf.variable_scope('layer4'):
            d['conv4'] = conv_layer(d['leaky_relu3'], 1, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm4'] = batchNormalization(d['conv4'], is_train)
            d['leaky_relu4'] = tf.nn.leaky_relu(d['batch_norm4'], alpha=0.1)
        # (104, 104, 128) --> (104, 104, 64)
        print('layer4.shape', d['leaky_relu4'].get_shape().as_list())

        #conv5 - batch_norm5 - leaky_relu5 - pool5
        with tf.variable_scope('layer5'):
            d['conv5'] = conv_layer(d['leaky_relu4'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm5'] = batchNormalization(d['conv5'], is_train)
            d['leaky_relu5'] = tf.nn.leaky_relu(d['batch_norm5'], alpha=0.1)
            d['pool5'] = max_pool(d['leaky_relu5'], 2, 2, padding='SAME')
        # (104, 104, 64) --> (52, 52, 128)
        print('layer5.shape', d['pool5'].get_shape().as_list())

        #conv6 - batch_norm6 - leaky_relu6
        with tf.variable_scope('layer6'):
            d['conv6'] = conv_layer(d['pool5'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm6'] = batchNormalization(d['conv6'], is_train)
            d['leaky_relu6'] = tf.nn.leaky_relu(d['batch_norm6'], alpha=0.1)
        # (52, 52, 128) --> (52, 52, 256)
        print('layer6.shape', d['leaky_relu6'].get_shape().as_list())

        #conv7 - batch_norm7 - leaky_relu7
        with tf.variable_scope('layer7'):
            d['conv7'] = conv_layer(d['leaky_relu6'], 1, 1, 128,
                                    padding='SAME', weights_stddev=0.01, biases_value=0.0)
            d['batch_norm7'] = batchNormalization(d['conv7'], is_train)
            d['leaky_relu7'] = tf.nn.leaky_relu(d['batch_norm7'], alpha=0.1)
        # (52, 52, 256) --> (52, 52, 128)
        print('layer7.shape', d['leaky_relu7'].get_shape().as_list())

        #conv8 - batch_norm8 - leaky_relu8 - pool8
        with tf.variable_scope('layer8'):
            d['conv8'] = conv_layer(d['leaky_relu7'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm8'] = batchNormalization(d['conv8'], is_train)
            d['leaky_relu8'] = tf.nn.leaky_relu(d['batch_norm8'], alpha=0.1)
            d['pool8'] = max_pool(d['leaky_relu8'], 2, 2, padding='SAME')
        # (52, 52, 128) --> (26, 26, 256)
        print('layer8.shape', d['pool8'].get_shape().as_list())

        #conv9 - batch_norm9 - leaky_relu9
        with tf.variable_scope('layer9'):
            d['conv9'] = conv_layer(d['pool8'], 3, 1, 512,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm9'] = batchNormalization(d['conv9'], is_train)
            d['leaky_relu9'] = tf.nn.leaky_relu(d['batch_norm9'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer9.shape', d['leaky_relu9'].get_shape().as_list())

        #conv10 - batch_norm10 - leaky_relu10
        with tf.variable_scope('layer10'):
            d['conv10'] = conv_layer(d['leaky_relu9'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm10'] = batchNormalization(d['conv10'], is_train)
            d['leaky_relu10'] = tf.nn.leaky_relu(d['batch_norm10'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer10.shape', d['leaky_relu10'].get_shape().as_list())

        #conv11 - batch_norm11 - leaky_relu11
        with tf.variable_scope('layer11'):
            d['conv11'] = conv_layer(d['leaky_relu10'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm11'] = batchNormalization(d['conv11'], is_train)
            d['leaky_relu11'] = tf.nn.leaky_relu(d['batch_norm11'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer11.shape', d['leaky_relu11'].get_shape().as_list())

        #conv12 - batch_norm12 - leaky_relu12
        with tf.variable_scope('layer12'):
            d['conv12'] = conv_layer(d['leaky_relu11'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm12'] = batchNormalization(d['conv12'], is_train)
            d['leaky_relu12'] = tf.nn.leaky_relu(d['batch_norm12'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer12.shape', d['leaky_relu12'].get_shape().as_list())

        #conv13 - batch_norm13 - leaky_relu13 - pool13
        with tf.variable_scope('layer13'):
            d['conv13'] = conv_layer(d['leaky_relu12'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm13'] = batchNormalization(d['conv13'], is_train)
            d['leaky_relu13'] = tf.nn.leaky_relu(d['batch_norm13'], alpha=0.1)
            d['pool13'] = max_pool(d['leaky_relu13'], 2, 2, padding='SAME')
        # (26, 26, 256) --> (13, 13, 512)
        print('layer13.shape', d['pool13'].get_shape().as_list())

        #conv14 - batch_norm14 - leaky_relu14
        with tf.variable_scope('layer14'):
            d['conv14'] = conv_layer(d['pool13'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm14'] = batchNormalization(d['conv14'], is_train)
            d['leaky_relu14'] = tf.nn.leaky_relu(d['batch_norm14'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer14.shape', d['leaky_relu14'].get_shape().as_list())

        #conv15 - batch_norm15 - leaky_relu15
        with tf.variable_scope('layer15'):
            d['conv15'] = conv_layer(d['leaky_relu14'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm15'] = batchNormalization(d['conv15'], is_train)
            d['leaky_relu15'] = tf.nn.leaky_relu(d['batch_norm15'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer15.shape', d['leaky_relu15'].get_shape().as_list())

        #conv16 - batch_norm16 - leaky_relu16
        with tf.variable_scope('layer16'):
            d['conv16'] = conv_layer(d['leaky_relu15'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm16'] = batchNormalization(d['conv16'], is_train)
            d['leaky_relu16'] = tf.nn.leaky_relu(d['batch_norm16'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer16.shape', d['leaky_relu16'].get_shape().as_list())

        #conv17 - batch_norm16 - leaky_relu17
        with tf.variable_scope('layer17'):
            d['conv17'] = conv_layer(d['leaky_relu16'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm17'] = batchNormalization(d['conv17'], is_train)
            d['leaky_relu17'] = tf.nn.leaky_relu(d['batch_norm17'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer17.shape', d['leaky_relu17'].get_shape().as_list())

        #conv18 - batch_norm18 - leaky_relu18
        with tf.variable_scope('layer18'):
            d['conv18'] = conv_layer(d['leaky_relu17'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm18'] = batchNormalization(d['conv18'], is_train)
            d['leaky_relu18'] = tf.nn.leaky_relu(d['batch_norm18'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer18.shape', d['leaky_relu18'].get_shape().as_list())

        #conv19 - batch_norm19 - leaky_relu19
        with tf.variable_scope('layer19'):
            d['conv19'] = conv_layer(d['leaky_relu18'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm19'] = batchNormalization(d['conv19'], is_train)
            d['leaky_relu19'] = tf.nn.leaky_relu(d['batch_norm19'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer19.shape', d['leaky_relu19'].get_shape().as_list())

        #conv20 - batch_norm20 - leaky_relu20
        with tf.variable_scope('layer20'):
            d['conv20'] = conv_layer(d['leaky_relu19'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm20'] = batchNormalization(d['conv20'], is_train)
            d['leaky_relu20'] = tf.nn.leaky_relu(d['batch_norm20'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer20.shape', d['leaky_relu20'].get_shape().as_list())

        # concatenate layer20 and layer 13 using space to depth
        with tf.variable_scope('layer21'):
            d['skip_connection'] = conv_layer(d['leaky_relu13'], 1, 1, 64,
                                              padding='SAME', use_bias=False, eights_stddev=0.01)
            d['skip_batch'] = batchNormalization(
                d['skip_connection'], is_train)
            d['skip_leaky_relu'] = tf.nn.leaky_relu(d['skip_batch'], alpha=0.1)
            d['skip_space_to_depth_x2'] = tf.space_to_depth(
                d['skip_leaky_relu'], block_size=2)
            d['concat21'] = tf.concat(
                [d['skip_space_to_depth_x2'], d['leaky_relu20']], axis=-1)
        # (13, 13, 1024) --> (13, 13, 256+1024)
        print('layer21.shape', d['concat21'].get_shape().as_list())

        #conv22 - batch_norm22 - leaky_relu22
        with tf.variable_scope('layer22'):
            d['conv22'] = conv_layer(d['concat21'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            d['batch_norm22'] = batchNormalization(d['conv22'], is_train)
            d['leaky_relu22'] = tf.nn.leaky_relu(d['batch_norm22'], alpha=0.1)
        # (13, 13, 1280) --> (13, 13, 1024)
        print('layer22.shape', d['leaky_relu22'].get_shape().as_list())

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['logit'] = conv_layer(d['leaky_relu22'], 1, 1, output_channel,
                                padding='SAME', use_bias=True, weights_stddev=0.01, biases_value=0.1)
        d['pred'] = tf.reshape(
            d['logit'], (-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + self.num_classes))
        print('pred.shape', d['pred'].get_shape().as_list())
        # (13, 13, 1024) --> (13, 13, num_anchors , (5 + num_classes))

        return d

    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성
        :param kwargs: dict, 추가 인자
                - loss_weights: list, 각 좌표, 신뢰도, 클래스 분류 확률에 대한 가중치 리스트
        :return tf.Tensor.
        """

        loss_weights = kwargs.pop('loss_weights', [5, 5, 5, 0.5, 1.0])
        # DEBUG
        # loss_weights = kwargs.pop('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
        grid_h, grid_w = self.grid_size
        num_classes = self.num_classes
        anchors = self.anchors
        grid_wh = np.reshape([grid_w, grid_h], [
                             1, 1, 1, 1, 2]).astype(np.float32)
        cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                             np.repeat(np.arange(grid_h), grid_w)])
        cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))

        txty, twth = self.pred[..., 0:2], self.pred[..., 2:4]
        confidence = tf.sigmoid(self.pred[..., 4:5])
        class_probs = tf.nn.softmax(
            self.pred[..., 5:], axis=-1) if num_classes > 1 else tf.sigmoid(self.pred[..., 5:])
        bxby = tf.sigmoid(txty) + cxcy
        pwph = np.reshape(anchors, (1, 1, 1, self.num_anchors, 2)) / 32
        bwbh = tf.exp(twth) * pwph

        # 예측(Prediction)을 위한 클래스 변수
        nxny, nwnh = bxby / grid_wh, bwbh / grid_wh
        nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh
        self.pred_y = tf.concat(
            (nx1ny1, nx2ny2, confidence, class_probs), axis=-1)

        # 각 앵커마다 IOU 계산
        num_objects = tf.reduce_sum(self.y[..., 4:5], axis=[1, 2, 3, 4])
        max_nx1ny1 = tf.maximum(self.y[..., 0:2], nx1ny1)
        min_nx2ny2 = tf.minimum(self.y[..., 2:4], nx2ny2)
        intersect_wh = tf.maximum(min_nx2ny2 - max_nx1ny1, 0.0)
        intersect_area = tf.reduce_prod(intersect_wh, axis=-1)
        intersect_area = tf.where(
            tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), intersect_area)
        gt_box_area = tf.reduce_prod(
            self.y[..., 2:4] - self.y[..., 0:2], axis=-1)
        box_area = tf.reduce_prod(nx2ny2 - nx1ny1, axis=-1)
        iou = tf.truediv(
            intersect_area, (gt_box_area + box_area - intersect_area))
        sum_iou = tf.reduce_sum(iou, axis=[1, 2, 3])
        self.iou = tf.truediv(sum_iou, num_objects)


        # 손실 함수를 위한 변수 생성 및 계산
        gt_bxby = 0.5 * (self.y[..., 0:2] + self.y[..., 2:4]) * grid_wh
        gt_bwbh = (self.y[..., 2:4] - self.y[..., 0:2]) * grid_wh

        resp_mask = self.y[..., 4:5]
        no_resp_mask = 1.0 - resp_mask
        gt_confidence = resp_mask * tf.expand_dims(iou, axis=-1)
        gt_class_probs = self.y[..., 5:]

        loss_bxby = loss_weights[0] * resp_mask * \
            tf.square(gt_bxby - bxby)
        loss_bwbh = loss_weights[1] * resp_mask * \
            tf.square(tf.sqrt(gt_bwbh) - tf.sqrt(bwbh))
        loss_resp_conf = loss_weights[2] * resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_no_resp_conf = loss_weights[3] * no_resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_class_probs = loss_weights[4] * resp_mask * \
            tf.square(gt_class_probs - class_probs)

       # 각 손실 함수 (xy, wh, confidence, no_confidence, class_probs)를 합친 뒤 평균내어 총 손실 함수 반환
        merged_loss = tf.concat((
                                loss_bxby,
                                loss_bwbh,
                                loss_resp_conf,
                                loss_no_resp_conf,
                                loss_class_probs
                                ),
                                axis=-1)
        total_loss = tf.reduce_sum(merged_loss, axis=-1)
        total_loss = tf.reduce_mean(total_loss)

        return total_loss

```

`YOLO` 클래스는 우선 YOLO에서 추가로 사용되는 변수를 생성합니다. Classification과 다르게 Detection에서 학습에서 사용되는 라벨(y)은 각 신경망마다 다른 경우가 많기 때문에 자식 클래스에서 구현하였습니다. 또 그리드의 크기와 앵커 개수 등을 추가로 생성하여 이후 손실 함수에 사용되게 하였습니다.

아키텍쳐 부분에서는 원 논문의 Darknet(layer1 ~ layer19) 및 YOLO Detector(layer20 ~ layer22)부분을 최대한 같게 구현하였습니다. 원 논문에서는 Darknet을 ImageNet Challenge 데이터셋으로 학습시켜 Feature extractor 성능을 끌어올렸으나, 이번 포스팅에선 그 부분을 생략하였습니다. 

손실 함수의 경우 Classification 때보다 다소 복잡합니다. dataset.data 모듈에서 설명한 것과 같이 책임을 갖는 앵커의 경우 검출할 객체의 좌표, 그 박스의 신뢰도(confidence), 클래스의 확률(class_probs)의 차이의 제곱을 총 손실로 가지며, 책임이 없는 앵커는 신뢰도가 0이 되게끔 손실을 줘 학습을 진행합니다. 제 부족한 설명으로 이해가 잘 안될 것 같은데요, 앞서 정의한 `_build_loss` 함수 부분과 밑의 손실 함수 수식을 차근차근 따라가면 이해에 큰 도움이 될 것 같습니다!

{% include image.html name=page.name file="yolo_loss_function.jpg" description="YOLO 신경망의 손실 함수 수식" class="large-image" %}

## (4) 러닝 알고리즘: SGD+Momentum

러닝 알고리즘은 Classification 문제때와 크게 다르지 않습니다. 마찬가지로 **모멘텀(momentum)**을 적용한 **확률적 경사 하강법(stochastic gradient descent; 이하 SGD)**을 채택하였으며, 베이스 클래스를 먼저 정의한 뒤, 이를 모멘텀 SGD에 기반한 optimizer 클래스가 상속받는 형태로 구현하였습니다.(추가로 Adam Opimizer 클래스도 구현해두었으니 관심이 있으신 분은 저장소에서 확인 후 이를 이용하여 학습도 해보시길 추천드립니다.)

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
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 100)
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
                - nms_flag: bool, nms(non maximum supression)를 수행할 지 여부.
        :return train_results: dict, 구체적인 학습 결과를 담은 dict
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())  # 전체 파라미터들을 초기화함

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
            step_loss, step_y_true, step_y_pred, step_X = self._step(
                sess, **kwargs)
            step_losses.append(step_loss)
            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i) % num_steps_per_epoch == 0:
                # 학습 데이터셋으로부터 추출한 현재의 미니배치에 대하여 모델의 예측 성능을 평가함
                step_score = self.evaluator.score(
                    step_y_true, step_y_pred, **kwargs)
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
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |lr: {:.6f}'
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

`Optimizer` 클래스에서 하는 일은 이전 Classification 문제와 거의 같기 때문에 추가로 설명은 하지 않겠습니다. 이전 포스팅의 자세한 설명을 참고해주세요! `MomentumOptimizer` 클래스 역시 거의 같지만 BatchNormalization 층을 학습하기 위하여 extra\_update\_ops를 추가로 정의하고 tf.control\_dependecies를 이용하여 학습을 시켜줍니다. 이를 정의하지 않으면 BatchNormalization 층이 학습이 되지 않아 전체 신경망이 제대로 작동하지 않을 가능성이 큽니다. 이외에는 모두 같습니다.



## 학습 수행 및 테스트 결과

`train.py` 스크립트에서 실제 학습을 수행하는 과정을 구현하며, `test.py` 스크립트에서 테스트 데이터셋에 대해 학습이 완료된 모델을 테스트하여 성능 수치를 보여주고 실제로 바운딩 박스도 그려줍니다. 혹, 레이블이 없는 데이터셋에 대해서 그려보고 싶은 분들을 위해 `draw.py` 스크립트도 추가 구현하였으니 저장소에서 참고하시길 바랍니다.

### train.py 스크립트

```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('data/face/') # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 앵커 로드
anchors = dataset.load_json(os.path.join(trainval_dir, 'anchors.json'))

# 학습에 사용될 이미지 사이즈 및 클래스 개수를 정함
IM_SIZE = (416, 416)
NUM_CLASSES = 1

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_data(trainval_dir, IM_SIZE)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.1) # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정"""
hp_d = dict()

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 2
hp_d['num_epochs'] = 50
hp_d['init_learning_rate'] = 1e-4
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
hp_d['nms_flag'] = True

""" 3. Graph 생성, session 초기화 및 학습 시작 """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))

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
root_dir = os.path.join('data/face')
test_dir = os.path.join(root_dir, 'test')
IM_SIZE = (416, 416)
NUM_CLASSES = 1

# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_data(test_dir, IM_SIZE)
test_set = dataset.DataSet(X_test, y_test)

""" 2. 테스트를 위한 하이퍼파라미터 설정 """
anchors = dataset.load_json(os.path.join(test_dir, 'anchors.json'))
class_map = dataset.load_json(os.path.join(test_dir, 'classes.json'))
nms_flag = True
hp_d = dict()

# FIXME
hp_d['batch_size'] = 16
hp_d['nms_flag'] = nms_flag

""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/tmp/model.ckpt')
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test performance: {}'.format(test_score))

""" 4. 이미지에 바운딩 박스 그리기 시작 """
draw_dir = os.path.join(test_dir, 'draws') # FIXME
im_dir = os.path.join(test_dir, 'images') # FIXME
im_paths = []
im_paths.extend(glob.glob(os.path.join(im_dir, '*.jpg')))
for idx, (img, y_pred, im_path) in enumerate(zip(test_set.images, test_y_pred, im_paths)):
    name = im_path.split('/')[-1]
    draw_path =os.path.join(draw_dir, name)
    if nms_flag:
        bboxes = predict_nms_boxes(y_pred, conf_thres=0.5, iou_thres=0.5)
    else:
        bboxes = convert_boxes(y_pred)
    bboxes = bboxes[np.nonzero(np.any(bboxes > 0, axis=1))]
    boxed_img = draw_pred_boxes(img, bboxes, class_map)
    cv2.imwrite(draw_path, boxed_img)
```
`test.py` 스크립트도 비슷하게 4단계 과정을 거쳐 성능을 측정하고 이미지에 예측된 오브젝트 바운딩 박스를 그려줍니다.

## 학습 결과 분석

### 학습 곡선

Classification 문제때와 마찬가지로 학습 수행 과정동안 학습 곡선을 그려보았습니다. 

{% include image.html name=page.name file="plot.png" description="학습 곡선 플롯팅 결과" class="large-image" %}

경험상으로 YOLO는 특별한 학습 정책을 주지않고 위의 방법대로 학습을 하는 경우, 배치사이즈가 어느정도 작을 때 더 학습이 잘 되는 것을 확인하여 학습 배치사이즈를 작게 가져가다보니 진동의 폭이 상대적으로 커 결과가 조금 알아보기 힘듭니다. 이에 편의상 재현율은 검증용 데이터셋에 대해만 표시하였습니다. 재현율 성능은 약 10 epoch 이후에 어느정도 수렴하기에, 이를 기반으로 최고 성능 모델을 저장하였습니다.

### 테스트 결과

테스트 결과 측정된 재현율(Recall)값은 **0.8734**로 꽤 높은 값을 가졌습니다. 하지만 재현율 값의 경우, 얼굴이 아닌 다른 곳을 얼굴이라고 예측해도 값에 지장을 주지 않기 때문에, 엄밀한 성능 지표라고 할 수 없습니다. 이후에 실험을 하실 때 mAP, F1 score, IOU 등 상황에 맞는 성능 지표를 만들어 학습을 진행하시면 보다 좋은 성능을 가지는 모델을 얻을 수 있습니다. 다시 돌아와서, 재현율 값만 가지고 모델이 정말 잘 예측하는 지 신뢰하기 힘들기 때문에 실제 테스트 이미지에서 정말 얼굴을 잘 인식했는지 예측된 바운딩 박스를 그려 눈으로 확인해보았습니다.

{% include image.html name=page.name file="correct.png" description="맞게 예측한 얼굴 인식 결과 예시" class="full-image" %}

{% include image.html name=page.name file="incorrect.png" description="잘못 예측한 얼굴 인식 결과 예시" class="full-image" %}

위의 결과 처럼 대부분 얼굴 부분을 잘 예측하지만, 손이나 얼굴의 일부 등을 얼굴로 예측하는 경우도 빈번하게 있었습니다. 이런 결과나온 이유 중에 하나로 실제 훈련데이터셋에서 확인해보면 얼굴의 형체가 거의 파악되지 않는 경우도 모두 annotation이 있어 그렇지 않나 조심스럽게 판단해봅니다. 하지만 대부분 꽤 잘 예측했기에 실용적으로도 사용할 수 있지 않을까 생각합니다!

## 결론

본 포스팅에서는 이미지 인식 분야에서 중요하게 다뤄지는 Detection 문제를 응용할 수 있는 **얼굴 인식**  사례를 소개하고 이를 YOLO 모델과 TensorFlow를 이용한 딥러닝 알고리즘으로 해결하는 과정을 간단하게 안내해드렸습니다. 미흡한 점이 많이 보이나, 실제 Detection 문제를 공부하고 처음 구현해보시는 분들에게 어느정도 도움이 되기를 바라며, 이해안되거나 바라는 점 댓글달아주시면 추후 보강하고 정리하여 다음 포스팅은 더 이해하기 쉽게 작성해보겠습니다.

**추후 글은 이미지 인식 분야 3가지 중 마지막 Segmentation에 대해 마찬가지로 예시 문제와 모델을 선정하여 해결하는 과정을 소개해드리겠습니다. 


## References

- YOLO 논문
  - <a href="https://arxiv.org/pdf/1612.08242.pdf" target="_blank">YOLO9000: Better, Faster, Stronger</a>
- FDDB dataset
  - <a href="http://vis-www.cs.umass.edu/fddb/samples/" target="_blank">FDDB: Face Detection Data Set and Benchmark</a>
