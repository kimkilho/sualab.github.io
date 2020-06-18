---
layout: post
title:  "Generative Adversarial Network : DCGAN을 이용한 이미지 생성"
date:   2019-05-08 09:00:00 +0900
author: hyunjun_kim
categories: [Introduction, Practice]
tags: [generative adversarial network, DCGAN, tensorflow]
comments: true
name: generative-adversarial-network
image: sample_image_random.jpg
---

안녕하세요. 이번에 포스팅 할 주제는 기존에 다루었던 내용들과는 조금 다른 내용을 이야기 해볼까 합니다. <a href="http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf" target="_blank">**Generative adversarial networks**(이하 GAN)</a>으로 불리는 분야에 대하여 소개 해 드리려 합니다. 해당 분야는 처음 등장한 2014년 이후로 지금까지 매우 많은 관심을 받으며 관련 연구도 많이 진행되었기에 하나에 포스트로 전부 다루는 것은 어려울 것으로 생각하여 가장 근간이 되는 **GAN**과 **DCGAN**에 대해서 소개하고 DCGAN의 경우는 코드와 함께 설명하고자 합니다.

* **다음과 같은 사항을 알고계시면 더 이해하기 쉽습니다.**
  - 딥러닝에 대한 전반적인 이해
  - Python 언어 및 TensorFlow 프레임워크에 대한 이해
* **GAN 방법론의 경우 random으로 sampling한 latent vector z에 의해 학습시 생성되는 이미지가 달라지므로 매번 같은 학습 결과가 나오지 않습니다.**
* 이번 글에서는 과거 구현체와 마찬가지로 데이터셋(data set), 성능 평가(performance evaluation), 러닝 모델(learning model), 러닝 알고리즘(leaning algorithm) 4가지 요소를 나눠 구현하였으며, 중복을 피하기 위해 다르게 구현한 부분 위주로 설명합니다. 
  - 전체 구현체 코드는 <a href="https://github.com/sualab/DCGAN_Face_gen_tf" target="_blank">수아랩의 GitHub 저장소</a>에서 자유롭게 확인하실 수 있습니다.
  - 데이터셋은 <a href="https://drive.google.com/file/d/1sR5PNAEWPmWGyEBYtprura4LsV_IUsYQ/view?usp=sharing" target="_blank">여기</a>서 받을 수 있습니다.
  - 성능평가에서 사용할 통계 데이터는 위의 수아랩 GitHub 저장소에서 구현체 코드를 통해 직접 계산할 수도 있지만 <a href="https://drive.google.com/file/d/14f5cQOCbiAoDODRmVmZvlNsg7wX0U92G/view?usp=sharing" target="_blank">여기</a>서 받을 수 있습니다.
  - 성능평가에서 사용할 pretrained Inception v3 그래프는 <a href="https://drive.google.com/file/d/1thIXF4jvG0KluzSEpsg1TCNiKEzh8VFV/view?usp=sharing" target="_blank">여기</a>서 받을 수 있습니다.
  

## 서론

GAN은 이미지를 생성하는 방법론으로 2014년 처음 등장한 이래로 매우 빠르게 연구되어 많은 방법론과 그 응용이 학계에 발표되었습니다. 단순히 이미지를 생성하는 수준에서 주어진 이미지를 다른 화풍의 이미지로 바꾸는 <a href="https://arxiv.org/pdf/1807.10201.pdf" target="_blank">style transfer</a>까지 다양한 응용이 가능하며 기존 <a href="https://arxiv.org/pdf/1511.06434.pdf" target="_blank">image recognition 작업의 성능을 향상</a>시킬 수도 있습니다.

{% include image.html name=page.name file="teaser_eccv18_cezanne.jpg" description="자동차 사진을 세잔 화풍으로 전이한 이미지" class="large-image" %}

GAN를 이해하기 위해서는 generative, adversarial 두가지 키워드에 대한 이해가 필요합니다. GAN에 대해 간략히 설명하자면 **adversarial learning**를 통해 **generative model**을 생성하는 방법론과 생성된 network를 의미합니다. 따라서 generative model은 어떤 학습 모델이며 adversarial learning는 어떤 방식인지를 알게 된다면 GAN이 어떤 분야인지 바로 이해하실 수 있습니다.

본 포스팅은 최대한 수식을 배제하고 일부의 필요한 수식만을 사용하여 설명하고자 합니다. 복잡한 수식이 사용되거나 이론적으로 깊은 이해가 필요한 부분에 대해서는 간략하게 언급하고 넘어갈 예정입니다.

### Generative Model

기존에 다루었던 image classification, object detection, image segmentation 문제들은 입력 이미지($$x$$)가 있을때 그에 따른 정답($$y$$)을 찾는 문제들입니다. image classification에서 주어진 이미지가 있을때, 그 이미지가 개의 이미지인지 고양이의 이미지인지 구별하는 문제등을 생각하면 됩니다. 이러한 모델은 **discriminative model**이라고 합니다. 즉 $$p(y \mid x)$$의 분포를 학습하여 이미지를 구별하는데 초점을 맞춘 모델이라고 생각하시면 됩니다. 일반적인 discriminative model의 경우 주어진 이미지를 구분하기 위한 특징점들을 찾아 분류하는 것을 목표로 하고 있습니다. 모델을 사람에 비유한다면 주어진 사진에서 보여지는 동물의 눈, 수염, 귀, 꼬리와 같은 요소를 보고 사진속의 동물이 개인지 고양이인지 구별하는 형태입니다.

{% include image.html name=page.name file="random-dogs-cats-predictions.png" description="개와 고양이를 구분하는 문제" class="large-image" %}

Dicriminative model을 사용하면서 한번쯤은 생각해봤을 상상이 있습니다. 만약 개의 눈, 수염, 귀, 꼬리 등의 요소가 어떻게 생겼는지를 알고 있으면 이를 이용해 개의 모습을 그릴 수 있지 않을까요? 긴 허리, 짧은 다리, 검은색과 갈색 털, 접힌 귀, 검은 눈동자를 위치에 맞게 그리면 닥스훈트 한마리를 그릴 수 있는 것입니다. 이 생각에서 착안한 것이 바로 **generative model**입니다. 즉, generative model은 데이터의 분포 $$p(x)$$를 학습하는 것을 목표로 하는 model을 의미합니다.

{% include image.html name=page.name file="MiniDachshund1_wb.jpg" description="긴 허리, 짧은 다리 등으로 우리는 닥스훈트를 인식 할 수 있다" class="large-image" %}
{% include image.html name=page.name file="drawn_dachshund.jpg" description="긴 허리, 짧은 다리 등을 그리면 닥스훈트를 그릴 수 있다." class="large-image" %}

일반적인 generative 모델은 discriminative 모델과 같이 오랜 기간 연구되어 왔습니다. 딥러닝이 보편화되기 이전에도 GMM(Gaussian Mixture Model), HMM(Hidden Markov Model)등의 방법론들을 중심으로 연구가 진행되어 왔습니다. 딥러닝이 보편화 된 이후 generative model은 **GAN(Generative Adversarial Networks)**, **VAE(Variational Auto-Encoder)**, 시계열 데이터 생성에 적합하다고 알려진 RNN(Recurrent Neural Network)등의 방향으로 연구되고 있습니다. 이 포스팅은 그 중 GAN에 대해서만 설명하며 다른 방법론들에 대해서는 기회가 된다면 나중에 다뤄보겠습니다.

{% include image.html name=page.name file="gen_models_anim_1.gif" description="VAE 학습 과정" class="large-image" %}
{% include image.html name=page.name file="GAN_samples.gif" description="GAN 학습 과정.(이번 구현으로 만들 수 있다.)" class="large-image" %}

### Adversarial Learning

**Adversarial learning**는 적대적이라는 단어에서 알 수 있듯이 두 개의 모델이 서로를 적대하며 학습하는 방식을 말합니다. 예를들어 두 모델을 각각 모델 A, 모델 B라고하면, 모델 A는 학습된 모델 B의 취약점을 찾아 교란하도록 학습하고 모델 B는 탐색된 취약점을 보완하는 방향으로 학습을 진행하는 방법론입니다. GAN에서 사용된 adversarial learning도 이와 비슷하게 진행됩니다.

## GAN

GAN은 이미지를 생성하기 위하여 2가지 모델을 동시에 사용합니다. **Generator model**과 **discriminator model**이 그것으로 두 모델은 서로에 대해 적대적인 관계를 가집니다. 자주 인용되는 비유로 **화폐 위조범과 경찰**이 있습니다. 화폐 위조범은 경찰의 눈을 속여 **가짜 화폐를 만들고** 경찰은 시중에 돌아다니는 **진짜 화폐와 가짜 화폐를 구분**합니다. 화폐 위조범은 적발을 피하기 위해 최대한 진짜 지폐와 유사하게 위폐를 만들 것이고, 경찰은 최대한 위폐와 진짜 화폐를 구별하기 위해 화폐의 여러 주요 포인트를 살펴볼 것입니다. 이러한 과정을 끝없이 반복하게 되면 **진짜 화폐와 똑같이 생긴 가짜 화폐를 만들게 되는 것**입니다.

GAN은 이러한 두 모델간의 경쟁을 discriminator $$D$$와 generator $$G$$ 사이의 **minimax game**으로 정의합니다. 둘 사이의 점수를 두고 한쪽($$G$$)은 점수를 최소로, 다른 한쪽($$D$$)은 점수를 최대로 하는 게임입니다. 일반적으로 경쟁을 하는 점수는 다음과 같이 표현합니다.

\begin{equation}
\DeclareMathOperator{\E}{\mathbb{E}}
\min_G \max_D V(D, G) = \E_{x \sim {p_{data}(x)}}[ \,\log(D(x))] \, +  \E_{z \sim {p_{z}(z)}}[ \, 1 - \log(D(G(z)))]\, 
\end{equation}

실제로 D, G를 각각 maximize, minimize하기보다 $$1 - \log(D(G(z)))$$ 대신 $$\log(D(G(z)))$$ 를 사용하여 해당 값을 maximize하는 방식으로 GAN을 학습하게 됩니다.

이 포스팅은 GAN을 직접 구현하는 것에 초점을 두고 있으므로 GAN의 수렴 여부 증명등 복잡한 수식을 설명하기 보다는 바로 코드를 통해 GAN에 대하여 설명하겠습니다. 처음에 등장한 GAN을 구현하는 것 보다는 조금 더 자주 쓰이는 DCGAN을 구현할 예정입니다. DCGAN은 GAN을 좀 더 개량한 논문으로 이후 등장하는 많은 GAN 논문들의 generator와 discriminator의 architecture를 구성하는데 많은 영감을 준 논문입니다.

## (1) Dataset: 얼굴 데이터셋 FFHQ

GAN을 학습하기 위해 사용하는 데이터로 **<a href="https://github.com/NVlabs/ffhq-dataset" target="_blank">Flickr Face HQ Dataset</a>** 을 사용하겠습니다. 이름에서 알 수 있듯이 Flickr를 통해 수집된 데이터들을 사용했으며 얼굴이미지만 모은 dataset이므로 별도의 annotation없이 학습이 진행됩니다. 실제로는 class정보를 이용하여 원하는 class의 데이터를 생성하는 것도 가능하지만 이번에 사용할 얼굴 데이터셋에서는 별도의 annotation이 없이 얼굴 이미지만 사용하여 GAN을 학습합니다.

{% include image.html name=page.name file="FFHQ_image_sample.jpg" description="FFHQ 데이터셋 예시" class="large-image" %}

데이터셋은 총 70000장으로 별도의 test나 evaluation용 데이터를 두지 않고 전부 학습에 사용합니다. 학습에는 64x64 크기의 이미지를 사용할 예정이므로 thumbnails128x128폴더의 데이터를 리사이즈 하여 사용합니다.

### datasets.data 모듈

`datasets.data`모듈은 데이터셋에 관련된 함수와 클래스를 가지고 있습니다. Classification 문제나 Detection, Segmentation 문제와 마찬가지로 이 모듈은 데이터셋을 메모리에 로드하고 학습 과정에서 이들을 미니 배치(minibatch) 단위로 제공합니다.

#### read_data 함수

```python
def read_data(data_dir, image_size, crop_size=None):
    """
    GAN을 학습하기 위해 데이터를 전처리하고 불러옴
    :param data_dir : str, image가 저장된 경로.
    :param image_size : tuple (width, height), 이미지를 resize할 경우 이미지 사이즈
    :param crop_size : int, 얼굴 이미지에서 배경을 제외한 얼굴만을 crop할경우 crop할 영역의 크기
    :return: X_set : np.ndarray, shape: (N, H, W, C).
    """
    img_list = [img for img in os.listdir(data_dir) if img.split(".")[-1] in IMAGE_EXTS]
    images = []
    
    for img in img_list:
        img_path = os.path.join(data_dir, img)
        im = imread(img_path)
        im = np.array(im, dtype=np.float32)
        if crop_size:
            im = center_crop(im, crop_size, crop_size)
        else:
            im = resize(im, (image_size[1], image_size[0]))
        im = im/127.5 - 1
        im = im[:,:,::-1]
        images.append(im)
        
    X_set = np.array(images, dtype=np.float32)
    
    return X_set
```

`read_data`함수는 데이터셋을 불러와 각 이미지를 crop하거나 resize하여 `numpy.ndarray` 형태로 변환합니다. crop을 하는 이유는 만약 생성하고 싶은 부분이 전체 이미지보다 작은 부분일 경우 해당 영역만을 잘라 generate를 도와주기 위함입니다. 또 generator에서 생성될 이미지 사이즈를 고려하여 resize 작업 역시 진행합니다. Generator의 작업을 원할하게 하기 위하여 학습 이미지를 **-1에서 1사이의 값으로 normalize**합니다. 또, GAN의 성능을 평가하기 위해 이미지의 채널 순서를 맞춰 주는 것이 중요합니다. 이 부분은 Evaluator를 설명하면서 같이 설명하겠습니다.

#### DataSet 클래스

```python
class Dataset(object):
    
    def __init__(self, images):
        """
        새로운 DataSet 객체를 생성함.
        :param images : np.ndarray, (N, H, W, C)
        """
        self._num_examples = images.shape[0]
        self._images = images
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()
        
    def _reset(self):
        """일부 변수를 재설정함."""
        self._epoch_completed = 0
        self._index_in_epoch = 0
        
    @property
    def images(self):
        return self._images
    
    @property
    def num_examples(self):
        return self._num_examples
    
    def next_batch(self, batch_size, shuffle=True):
        """
        `batch_size` 개수만큼의 이미지들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 반환함.
        :param batch_size : int, 미니배치 크기.
        :param shuffle : bool, 미치배치 추출에 앞서, 현재 데이터셋 내 이미지들의 순서를 랜덤하게 섞을 것인지 여부.
        :return: batch_images : np.ndarray, shape: (N,H,W,C)
        """
        
        start_index = self._index_in_epoch
        
        if self._epoch_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)
        
        if start_index + batch_size > self._num_examples:
            self._epoch_completed += 1
            rest_num_examples = self._num_examples - start_index
            
            indices_rest_part = self._indices[start_index:self._num_examples]
            
            if shuffle:
                np.random.shuffle(self._indices)
            
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]
            
            images_rest_part = self._images[indices_rest_part]
            images_new_part = self._images[indices_new_part]
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self._images[indices]
        
        return batch_images
```

`Dataset` 클래스를 이용하여 메모리에 로드된 `X_set`을 미니배치(minibatch) 단위로 반환합니다.

## (2) 성능 평가 : Fréchet Inception Distance

GAN의 성능을 평가하는 것은 완벽하지 않습니다. 초창기에는 생성된 이미지를 정성적으로 평가하는 방식으로 진행하였고 그 후에 **<a href="https://arxiv.org/pdf/1606.03498.pdf" target="_blank">Inception Score(IS)</a>** 가 등장하면서 별도의 네트워크를 이용하여 생성된 이미지의 성능을 평가하기 시작하였습니다. 이 포스팅에서 사용할 성능 평가지표는 **<a href="https://arxiv.org/pdf/1706.08500.pdf" target="_blank">Fréchet Inception Distance(FID)</a>** 입니다. 두 지표의 이름에서 눈치채셨을 지도 모르겠지만 두 지표 모두 Inception network를 사용하여 성능을 측정합니다.

FID는 간단하게 요약하면 **real data와 fake data의 feature space상에서의 거리**입니다. **Inception network**(Inception V3을 주로 사용합니다.) 를 이용하여 real data와 fake data의 feature를 추출한 뒤, 두 집합의 feature의 **mean과 covariance $$(m_r,C_r), (m_f,C_f)$$** 를 구한뒤 각 값을 이용하여 거리를 계산합니다. 계산식은 다음과 같습니다.

\begin{equation}
\DeclareMathOperator{\Tr}{Tr}
FID^2 = ||m_f - m_r||^2_2 + \Tr(C_f + C_r - 2(C_f C_r)^{1/2})
\end{equation}

### learning.fid 모듈

FID Evaluator를 구현하기 위해 먼저 `FID` 클래스를 구현합니다. 이는 `learning.fid` 모듈에 구현하였습니다.

#### FID 클래스

```python
class FID(object):
    """Frechet Inception Distance 를 계산하기 위한 클래스."""
    def __init__(self, model_path, dataset_stats_path, sess):
        """
        새로운 FID 객체를 생성함.
        :param model_path : str, FID를 계산하는 Inception model(*.pb) 파일의 경로.
        :param dataset_path : Dataset object, m_w 와 C_w 를 계산할 데이터셋.
        :param sess : tf.Session, using inception network를 이용하여 피쳐를 추출하는 세션.
        """
        self.inception_layer = self.get_inception_layer(sess, model_path)
        self.mu_data, self.sigma_data = self.get_data_stats(dataset_stats_path)
        self.sess = sess
        # 2048 은 inception network의 피쳐크기.
        self.feature_gen = np.empty((0, 2048))
		
		
    def get_inception_layer(self, sess, model_path):
        try:
            pool_layer = sess.graph.get_tensor_by_name(
                "FID/InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0")
        except KeyError:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name="FID")
            pool_layer = sess.graph.get_tensor_by_name(
                "FID/InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0")
        
        ops = pool_layer.graph.get_operations()
        
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != [] and (shape._dims is not None):
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        
        return pool_layer
    
    def get_data_stats(self, dataset_stats_path):
        assert os.path.exists(dataset_stats_path)
        
        with open(dataset_stats_path, 'rb') as f:
            stats = pkl.load(f)
        
        return stats["mu"], stats["sigma"]
    
    def reset_FID(self):
        self.feature_gen = np.empty((0, 2048))
    
    def extract_inception_features(self, images):
        batch_size = images.shape[0]
        images = (images+1) * 127.5
        pred = self.sess.run(self.inception_layer, {'FID/input:0' : images})
        self.feature_gen = np.append(self.feature_gen,
                                     pred.reshape(batch_size, -1), axis=0)
    
    def calculate_FID(self):
        pred_arr = self.feature_gen
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        
        if mu.shape != self.mu_data.shape:
            print("shape of mu is {}, shape of mu_data is {}".format(
                mu.shape, self.mu_data.shape))
        
        assert mu.shape == self.mu_data.shape, "Two means have different lengths"
        assert sigma.shape == self.sigma_data.shape, "Tow cov have different size"

        diff = mu - self.mu_data

        cov_mean, _ = linalg.sqrtm(sigma.dot(self.sigma_data), disp=False)
        if not np.isfinite(cov_mean).all():
            print("Singular product has happened when calculate FID. adding \
                  %s to diagonal of cov estimates" % 1e-6)
            offset = np.eye(sigma.shape[0]) * 1e-6
            cov_mean = linalg.sqrtm((sigma + offset).dot(self.sigma_data + offset))
            
        if np.iscomplexobj(cov_mean):
            if not np.allclose(np.diagonal(cov_mean).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_mean.imag))
                raise ValueError("Imaginary component {}".format(m))
            cov_mean = cov_mean.real

        return np.sqrt(diff.dot(diff) + np.trace(sigma) + np.trace(self.sigma_data) 
                       - 2 * np.trace(cov_mean))
```

Pretrain된 Inception V3 network는 직접 구하셔도 되지만 편의를 위해 <a href="https://drive.google.com/file/d/1thIXF4jvG0KluzSEpsg1TCNiKEzh8VFV/view?usp=sharing" target="_blank">여기</a>에서 받는 것을 추천드립니다. 또 FFHQ data의 feature mean, feature covariance는 미리 계산하여 <a href="https://drive.google.com/file/d/14f5cQOCbiAoDODRmVmZvlNsg7wX0U92G/view?usp=sharing" target="_blank">여기</a>에 링크해 놓았으니 직접 계산하셔도 되고 받아서 사용하셔도 됩니다. `extract_inception_features`에서 -1에서 1사이의 값을 0부터 255사이의 값으로 바꾸어 Inception network에서 feature를 추출하였고 제공하는 Inception Network가 **rgb의 channel순서**로 입력 이미지가 구성되어있으므로 **채널의 순서를 유의하여 주시기 바랍니다**. Fake data의 mean을 구하기 위해서는 충분한 수의 sample이 있어야 하므로 학습시의 메모리를 고려하여 batch단위로 feature를 뽑도록 하였습니다. 이는 이후 `Evaluator` 클래스를 구현하는데 있어 `FID` 클래스를 별도로 구현한 이유이기도 합니다.

### learning.evaluator 모듈

`learning.evaluator` 모듈은 현재까지 학습된 모델의 성능 평가를 위한 'evaluator'의 클래스를 담고 있습니다. `Evaluator` 클래스는 <a href="http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html" target="_blank">image classification 포스팅</a>등에서 이미 구현한 바 있으므로 생략하겠습니다.

#### FIDEvaluator 클래스

```python
class FIDEvaluator(Evaluator):
    """FID score를 평가 척도로 사용하는 evaluator 클래스."""
    
    @property
    def worst_score(self):
        """최악의 성능 점수."""
        return 1000.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부."""
        return 'min'

    def score(self, sess, fid, model,**kwargs):
        """FID에 기반한 성능 평가점수."""
        batch_size_eval = kwargs.pop('batch_size_eval', 50)
        eval_sample_size = kwargs.pop('eval_sample_size', 10000)
        n_iter = eval_sample_size // batch_size_eval
        fid.reset_FID()
        for i in range(n_iter):
            z_eval = np.random.uniform(-1.0, 1.0, size=(batch_size_eval, model.z_dim))
            .astype(np.float32)
            eval_generated = model.generate(sess, z_eval, verbose=False, **kwargs)
            fid.extract_inception_features(eval_generated)
        score = fid.calculate_FID()
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
        relative_eps = 1.0 - score_threshold
        return curr < best * relative_eps
```

`Evaluator` 클래스를 상속받아 `FIDEvlauator` 클래스를 구현하였습니다. 학습된 model의 generate 함수를 통해 fake image를 생성하고 이를 `FID` 클래스의 `extract_inception_features`함수를 통해 feature를 추출합니다. evaluate를 위한 sample의 사이즈가 커 한번에 feature를 뽑을 수 없기에 minibatch로 나누어 진행합니다. 이후 `calculate_FID` 함수를 통해 FID값을 계산합니다.
낮을 수록 좋은 성능 척도이므로 `mode`를 'min'으로 `score_threshold` 값을 1e-4로 설정하였습니다.

## (3) 러닝 모델: DCGAN (Deep Convolution Generative Adversarial Network)

러닝 모델로는 앞서 언급한대로 DCGAN을 사용합니다. 다른 포스팅과 마찬가지로 주로 사용하는 층(layer)들을 생성하는 함수를 `models.layers`에서 먼저 정의하고 `models.nn`모듈에서 일반적인 GAN 모델을 정의하고 이를 DCGAN이 상속 받도록 구현하였습니다.

### models.layers 모듈

```python
def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias,
                             kernel_initializer=tf.initializers.random_normal(0.0, 0.02))

def deconv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, use_bias=use_bias, 
                                      kernel_initializer=tf.initializers.random_normal(0.0, 0.02))

def batchNormalization(x, is_train):
    """
    새로운 batchNormalization 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), True이면 train mode, 아니면 test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.9, epsilon=1e-5, 
                                        center=True, scale=True)


def conv_bn_lrelu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', bn=True, alpha=0.2):
    """
    conv + bn + Leaky Relu 으로 이루어진 층을 추가함.
    conv_layer, batchNormalization 함수 참고.
    relu를 사용하고 싶으면, alpha를 0으로 설정.
    activation 층을 사용하고 싶지 않으면 alpha를 1.0으로 설정.
    """
    conv = conv_layer(x, filters, kernel_size, strides, padding, use_bias=True)
    if bn:
        _bn = batchNormalization(conv, is_train)
    else:
        _bn = conv
    return tf.nn.leaky_relu(_bn, alpha)
    
def deconv_bn_relu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', bn=True, relu=True):
    """
    deconv + bn + Relu 으로 이루어진 층을 추가함.
    deconv_layer, batchNormalization 함수 참고.
    """
    deconv = deconv_layer(x, filters, kernel_size, strides, padding, use_bias=True)
    if bn:
        _bn = batchNormalization(deconv, is_train)
    else:
        _bn = deconv
    if relu:
        return tf.nn.relu(_bn)
    else:
        return _bn


def fc_layer(x, out_dim, **kwargs):
    """
    새로운 완전 연결 층을 추가함.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, 출력 벡터의 차원수.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.02)
    biases_value = kwargs.pop('biases_value', 0.0)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases


def fc_bn_lrelu(x, out_dim, is_train, alpha=0.2):
    """
    fc + bn + Leaky Relu 으로 이루어진 층을 추가함.
    fc_layer, batchNormalization 함수 참고.
    """
    fc = fc_layer(x, out_dim)
    bn = batchNormalization(fc, is_train)
    return tf.nn.leaky_relu(bn, alpha)
```

`models.layers` 모듈은 신경망을 구성하는데 필요한 layer들에 대해 정의한 모듈입니다. tf.layer 모듈을 이용하여 간편하게 정의하였으며 기존에 사용하던 ReLU이외에 LeakyReLU를 사용한 layer도 구현하였습니다.
LeakyReLU는 ReLU와 비슷하지만 입력이 음수일경우 0을 내보내는 것이 아니라 일정 비율을 입력에 곱한 값을 출력으로 내보내는 함수입니다. DCGAN에서는 discriminator를 구성하는데 사용됩니다. fc_layer는 image classification에서 사용한 함수를 그대로 사용하였습니다.

### models.nn 모듈

`models.nn` 모듈은 신경망을 표현하는 클래스를 가지고 있습니다.

#### GAN 클래스

```python
class GAN(metaclass=ABCMeta):
    """Generative Adversarial Network의 베이스 클래스."""
    
    def __init__(self, input_shape, **kwargs):
        """
        모델을 초기화한다.
        :param input_shape: np.array, shape [H,W,C]
        """

        if input_shape is None:
            input_shape = [None, None, 3]
        self.z_dim = kwargs.pop('z_dim', 100)
        self.c_dim = input_shape[-1]
            
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.z = tf.placeholder(tf.float32, [None] + [self.z_dim])
        self.is_train = tf.placeholder(tf.bool)
        
        self.G = self._build_generator(**kwargs)
        self.D, self.D_logits, self.D_l4 = self._build_discriminator(False, **kwargs)
        self.D_, self.D_logits_, _ = self._build_discriminator(True, **kwargs)
        self.G_ = self._build_sampler(**kwargs)
        
        self.gen_loss, self.discr_loss = self._build_loss(**kwargs)
        
    @abstractmethod
    def _build_generator(self, **kwargs):
        """
        Generator를 빌드.
        해당 함수를 추후 구현해야 함.
        """
        pass
    
    @abstractmethod
    def _build_sampler(self, **kwargs):
        """
        Sampler를 빌드.
        해당 함수를 추후 구현해야 함.
        """
        pass
    
    @abstractmethod
    def _build_discriminator(self, **kwargs):
        """
        Discriminator를 빌드.
        해당 함수를 추후 구현해야 함.
        """
        pass
    
    
    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        generator 와 discriminator를 위한 로스를 반환함.
        해당 함수를 추후 구현해야 함.
        """
        pass
    
    def generate(self, sess, z, verbose=False, **kwargs):
        """
        z 벡터를 이용해서 이미지를 생성함.
        :param sess: tf.Session
        :param z: np.ndarray, (N, z_dim)
        :param verbose: bool, 생성 과정에서 구체적인 정보를 출력할 것인지 여부.
        :params kwargs: dict, 생성을 위한 추가 인자.
                -batch_size: int, 각 반복 회차에서의 미니배치 크기.
        :return _image_gen: np.ndarray, shape: shape of (N, H, W, C)
        """
        
        batch_size = kwargs.pop('batch_size', 64)
        
        num_image = z.shape[0]
        num_steps = num_image//batch_size
        
        if verbose:
            print("Running generation loop...")
        
        
        _image_gen = []
        start_time = time.time()
        for i in range(num_steps + 1):
            start_batch = i * batch_size
            
            if i==num_steps:
                _batch_size = num_image - num_steps * batch_size
            else:
                _batch_size = batch_size
            
            end_batch = start_batch + _batch_size
            z_batch = z[start_batch:end_batch]
            
            image_gen = sess.run(self.G_, feed_dict={
                                self.z : z_batch, self.is_train: False})
            _image_gen.append(image_gen)
            
        if verbose:
            print('Total generation time(sec): {}'.format(
                time.time() - start_time))
        
        _image_gen = np.concatenate(_image_gen, axis=0)
        
        return _image_gen
```

`GAN` 클래스는 기본 추상 베이스 클래스로, 확장성을 위해 전반적인 GAN을 포괄하도록 구현하였습니다. `_build_generator`, `_build_discriminator`, `_build_sampler`, `_build_loss` 함수는 `GAN`의 자식 클래스에서 구현하도록 하였고, `generate` 함수는 학습한 generator에서 이미지를 생성합니다. `_build_sampler` 함수는 기본적으로 `_build_generator` 함수와 같지만 학습에 직접적으로 사용하는 것이 아니라 evaluation이나 test 단계에서 사용할 수 있도록 별도의 함수로 구성하였습니다. 또한 **$$z$$** 라고 하는 placeholder를 정의한 것을 확인 할 수 있습니다. 이는 GAN에서 이미지를 생성하는데 중요한 입력으로 **학습이후에 이미지를 생성할때 이미지의 속성을 결정할 vector**가 됩니다.

#### DCGAN 클래스

```python
class DCGAN(GAN):
    """
    DCGAN 클래스
    see: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    https://arxiv.org/abs/1511.06434
    """
    def _build_generator(self, **kwargs):
        """
        generator 생성.
        :param kwargs: dict, generator 생성을 위한 추가 인자.
        :return tf.Tensor
        """
        d = dict()
        c_dim = self.X.shape[-1]
        kernel_size = (5,5)
        fc_channel = kwargs.pop('G_FC_layer_channel', 1024)
        G_channel = kwargs.pop('G_channel', 64)
        
        with tf.variable_scope("generator") as scope:
            z_input = self.z
            
            d['layer_1'] = fc_layer(z_input, 4*4*fc_channel)
            d['reshape'] = tf.nn.relu(batchNormalization(tf.reshape(d['layer_1'], [-1, 4, 4, fc_channel]), self.is_train))
            d['layer_2'] = deconv_bn_relu(d['reshape'], G_channel*4, kernel_size, self.is_train, strides=(2,2))
            d['layer_3'] = deconv_bn_relu(d['layer_2'], G_channel*2, kernel_size, self.is_train, strides=(2,2))
            d['layer_4'] = deconv_bn_relu(d['layer_3'], G_channel, kernel_size, self.is_train, strides=(2,2))
            d['layer_5'] = deconv_bn_relu(d['layer_4'], c_dim, kernel_size, self.is_train, strides=(2,2), bn=False, relu=False)
            d['tanh'] = tf.nn.tanh(d['layer_5'])
            
        return d['tanh']
	
    def _build_sampler(self, **kwargs):
        """
        sampler 생성.
        :param kwargs: dict, sampler 생성을 위한 추가 인자.
        :return tf.Tensor
        """
        d = dict()
        c_dim = self.X.shape[-1]
        kernel_size = (5,5)
        fc_channel = kwargs.pop('G_FC_layer_channel', 1024)
        G_channel = kwargs.pop('G_channel', 64)
        
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            z_input = self.z
            
            d['layer_1'] = fc_layer(z_input, 4*4*fc_channel)
            d['reshape'] = tf.nn.relu(batchNormalization(tf.reshape(d['layer_1'], [-1, 4, 4, fc_channel]), self.is_train))
            d['layer_2'] = deconv_bn_relu(d['reshape'], G_channel*4, kernel_size, self.is_train, strides=(2,2))
            d['layer_3'] = deconv_bn_relu(d['layer_2'], G_channel*2, kernel_size, self.is_train, strides=(2,2))
            d['layer_4'] = deconv_bn_relu(d['layer_3'], G_channel, kernel_size, self.is_train, strides=(2,2))
            d['layer_5'] = deconv_bn_relu(d['layer_4'], c_dim, kernel_size, self.is_train, strides=(2,2), bn=False, relu=False)
            d['tanh'] = tf.nn.tanh(d['layer_5'])
            
        return d['tanh']
    
    def _build_discriminator(self, fake_image=False, **kwargs):
        """
        discriminator 생성.
        :param fake_images: bool, 생성한 가상 이미지인지 여부.
        :param kwargs: dict, discriminator 생성을 위한 추가 인자.
        :return (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        d = dict()
        kernel_size = (5,5)
        if fake_image:
            input_image = self.G
        else:
            input_image = self.X
        batch_size = kwargs.pop('batch_size', 8)
        D_channel = kwargs.pop('D_channel', 64)
        
        with tf.variable_scope("discriminator") as scope:
            if fake_image:
                scope.reuse_variables()
            
            d['layer_1'] = conv_bn_lrelu(input_image, D_channel, kernel_size, self.is_train, strides=(2,2), bn=False)
            d['layer_2'] = conv_bn_lrelu(d['layer_1'], D_channel*2, kernel_size, self.is_train, strides=(2,2))
            d['layer_3'] = conv_bn_lrelu(d['layer_2'], D_channel*4, kernel_size, self.is_train, strides=(2,2))
            d['layer_4'] = conv_bn_lrelu(d['layer_3'], D_channel*8, kernel_size, self.is_train, strides=(2,2))
            d['layer_5'] = fc_layer(tf.contrib.layers.flatten(d['layer_4']),1)
            d['sigmoid'] = tf.nn.sigmoid(d['layer_5'])
			
        return d['sigmoid'], d['layer_5'], d['layer_1']
    
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성
        :return tf.Tensor
        """
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        
        d_loss = d_loss_real + d_loss_fake
        return g_loss, d_loss
```

`_build_generator` 함수는 임의의 random vector $$z$$로부터 이미지를 생성하는 네트워크를 구성합니다. DCGAN 논문이 많은 영향을 준 부분이 이 부분입니다. 초창기 GAN에서는 generator를 구성할때 주로 Fully-connected layer를 사용하였지만 DCGAN이라는 이름에서 알 수 있듯이 이 논문에서 **Convolution layer** 를 사용하게 됩니다. Generator에서 사용하는 conv layer는 편의상 deconv로 표기 했지만 정식 명칭은 **fractionally-strided convolution**으로 strided convolution이 일반적으로 이미지의 크기를 줄여주는 것과 반대 역할을 하게 됩니다.


{% include image.html name=page.name file="DCGAN_gen.JPG" description="DCGAN에서 사용한 Generator 구조. 사용한 Dataset이 FFHQ와는 다르므로 일부 값이 변경됨" class="full-image" %}

`_build_sampler`함수는 기본적으로 generator와 같은 network를 사용하는 것을 목적으로 생성합니다. variable을 reuse하여 별도의 망을 만드는 것이 아닌 `_build_generator`로 생성된 망을 사용하게 합니다.

`_build_discriminator` 함수는 `_build_generator`에서 생성된 이미지나 학습에 사용된 real 이미지를 입력으로 받아 이 이미지가 real인지 fake인지 1-bit의 logit으로 출력하도록 하는 망을 생성합니다. real이면 1, fake면 0이 나오는 망입니다. fake image와 real image에서의 logit이 다르게 나와야 하므로 `GAN` 클래스에서 `self.D_`이나 `self.D_logits_`라는 별도의 변수를 통해 fake image에 대한 network을 구성합니다.

DCGAN 논문에서 제안하는 바는 사용하는 batch normalization과 activation layer에도 있습니다. `_build_generator`의 마지막 layer와 `_build_discriminator`의 첫 layer에는 batch normalization을 사용하지 않았습니다. 또, activation layer의 경우 `_build_generator`에서는 마지막 layer에 tanh를 적용한 것 이외에 전부 ReLU를 사용하였고, `_build_discriminator`에서는 LeakyReLU를 사용하였습니다.

`_build_loss`함수는 손실함수를 구현하였습니다. 기본적인 손실 함수 자체는 GAN에서 제안한 식을 그대로 사용하였습니다. 먼저 discriminator loss는 real image에서 1이 나와야 하고 fake image에서 0이 나와야 합니다. 따라서 `d_loss_real`은 1 bit logit이 1이 나오도록 sigmoid cross-entropy loss를 구성하였고, `d_loss_fake`는 1 bit logit이 0이 나오도록 sigmoid cross-entropy loss를 구성하였습니다. 그리고 discriminator loss는 두 loss의 합이 됩니다. 한편 `g_loss`는 오직 fake image를 discriminator에 넣었을때 1 bit logit이 1이 나와야 하므로 `d_loss_fake`와는 반대로 sigmoid cross-entropy loss를 구성하였습니다. Discriminator, generator는 따로 학습을 해야 하므로 두 loss를 합하지 않고 별도로 저장합니다.

## (4) 러닝 알고리즘 : SGD+Momentum

러닝 알고리즘은 앞서 다룬 문제들과 크게 다르지 않습니다. **모멘텀(momentum)**을 적용한 **확률적 경사 하강법(stochastic gradient descent; 이하 SGD)**을 채택하였으며, 베이스 클래스를 먼저 정의한 뒤, 이를 모멘텀 SGD에 기반한 optimizer 클래스가 상속받는 형태로 구현하였습니다. Pretrained model weights를 불러오는 부분을 제외하고 <a href="http://research.sualab.com/practice/2018/05/14/image-detection-deep-learning.html" target="_blank">Detection 포스팅</a>때와 동일하니 설명은 생략하도록 하겠습니다.

### learning.optimizer 모듈

#### Optimizer 클래스

```python
class Optimizer(metaclass=ABCMeta):
    """경사 하강 러닝 알고리즘 기반 optimizer의 베이스 클래스"""

    def __init__(self, model, train_set, evaluator, **kwargs):
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
        self.sample_H = kwargs.pop('sample_H', 2)
        self.sample_W = kwargs.pop('sample_W', 10)
        z_dim = kwargs.pop('z_dim', 100)
        self.sample_z = self.z_sample(self.sample_H, self.sample_W, z_dim)

        # 학습 하이퍼파라미터
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 100)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.001)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
           
        self.optimize_G = self._optimize_op("generator")
        self.optimize_D = self._optimize_op("discriminator")

        self._reset()

    def _reset(self):
        """일부 변수를 재설정."""
        self.curr_epoch = 1
        # 'bad epochs' 수: 성능 향상이 연속적으로 이루어지지 않은 epochs 수.
        self.num_bad_epochs = 0
        # 최저 성능 점수로, 현 최고 점수를 초기화함.
        self.best_score = self.evaluator.worst_score	
        self.curr_learning_rate = self.init_learning_rate
       
    def z_sample(self, H, W, z_dim):
        z = np.random.uniform(-1.0, 1.0, size=(W*H, z_dim))
        return z


    @abstractmethod
    def _optimize_op(self, mode, **kwargs):
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
        :return generator loss: float, 1회 반복 회차 결과 gnerator의 손실 함수값.
                dicriminator loss: float, 1회 반복 회차 결과 discriminator의 손실 함수값.
                X: np.ndarray, 학습 데이터셋의 실제 이미지.
                G: np.ndarray, 모델이 생성한 이미지.
        """

        # 미니배치 하나를 추출함
        X = self.train_set.next_batch(self.batch_size, shuffle=True)
        z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, 
                                               self.model.z_dim)).astype(np.float32)
        # 손실 함숫값을 계산하고, 모델 업데이트를 수행함
        # Generator는 두번 업데이트 됨.
        _, D_loss = \
            sess.run([self.optimize_D, self.model.discr_loss, self.model.D_l4],
                feed_dict={self.model.z: z, self.model.X: X, self.model.is_train: True, 
                           self.learning_rate_placeholder: self.curr_learning_rate})
        _, G_loss, G = \
            sess.run([self.optimize_G, self.model.gen_loss, self.model.G],
                feed_dict={self.model.z: z, self.model.X: X, self.model.is_train: True, 
                           self.learning_rate_placeholder: self.curr_learning_rate})
        _, G_loss, G = \
            sess.run([self.optimize_G, self.model.gen_loss, self.model.G],
                feed_dict={self.model.z: z, self.model.X: X, self.model.is_train: True, 
                           self.learning_rate_placeholder: self.curr_learning_rate})
        return G_loss, D_loss, X, G
    
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
        sess.run(tf.global_variables_initializer())	# 모든 파라미터들을 초기화.
        
        inception_path = kwargs.pop('inception_path', 
                                    './inception/inception-2015-12-05/ \
                                    classify_image_graph_def.pb')
        dataset_stats_path = kwargs.pop('dataset_stats_path', 
                                        './data/thumbnails128x128/stats.pkl')
        fid = FID(inception_path, dataset_stats_path, sess)

        train_results = dict()
        train_size = self.train_set.num_examples
        print("Size of train set :", train_size)
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        n_eval = kwargs.pop('eval_sample_size',10000)
        batch_size_eval = kwargs.pop('batch_size_eval',500)
        
        sample_dir = kwargs.pop('sample_dir', save_dir)
        
        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses_G, step_losses_D, step_scores, eval_scores = [], [], [], []
        start_time = time.time()

        # 학습 루프를 실행함.
        for i in range(num_steps):
            # 미니배치 하나로부터 경사 하강 업데이트를 1회 수행함
            step_loss_G, step_loss_D, step_X, gen_img, D = self._step(sess, **kwargs)
            step_losses_G.append(step_loss_G)
            step_losses_D.append(step_loss_D)
            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i) % 10 == 0:
                print('[step {}]\tG_loss: {:.6f}|D_loss:{:.6f} |lr: {:.6f}'\
                      .format(i, step_loss_G, step_loss_D, self.curr_learning_rate))
            if (i) % num_steps_per_epoch == num_steps_per_epoch - 1:
                # 학습셋에서 추출한 현재 미니배치로 모델을 평가함.
                fid.reset_FID()
                fid.extract_inception_features(gen_img)
                step_score = fid.calculate_FID()
                step_scores.append(step_score)

                sample_image = self.model.generate(sess, self.sample_z, verbose=False, 
                                                   **kwargs)
                   
                save_sample_images(sample_dir, i, sample_image, self.sample_H, 
                                   self.sample_W)
                                
                eval_score = self.evaluator.score(sess, fid, self.model, **kwargs)
                eval_scores.append(eval_score)

                if verbose:
                    # 중간 결과 출력.
                    print('[epoch {}]\tG_loss: {:.6f}|D_loss:{:.6f} |Train score: {:.6f} \
                    |Eval score: {:.6f} |lr: {:.6f}'\
                        .format(self.curr_epoch, step_loss_G, step_loss_D, step_score, 
                                eval_score, self.curr_learning_rate))
                    # 중간 결과 플롯팅함.
                    plot_learning_curve(-1, step_losses_G, step_losses_D, step_scores, 
                                        eval_scores=eval_scores, img_dir=save_dir)

                curr_score = eval_score

                # 현재의 성능 점수의 현재까지의 최고 성능 점수를 비교하고, 
                # 최고 성능 점수가 갱신된 경우 해당 성능을 발휘한 모델의 파라미터들을 저장함
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, 
                               os.path.join(save_dir, 
                                            'model_{}.ckpt'.format(self.curr_epoch)))
                else:
                    self.num_bad_epochs += 1

# 			    self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training', 
                                             self.best_score))

        print('Done.')

        if details:
            # 모델 저장.
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
            # 학습 결과를 dict에 저장함.
            train_results['step_losses_G'] = step_losses_G
            train_results['step_losses_D'] = step_losses_D
            train_results['step_scores'] = step_scores
            train_results['eval_scores'] = eval_scores

            return train_results
```

`Optimizer` 클래스에서 주목해야 할 부분은 `_step` 함수입니다. 일반적으로 discriminator와 generator의 학습을 동등하게 진행되면 좋지만 일반적으로 discriminator가 먼저 학습되는 경우가 많아 **generator를 한번에 두번씩 학습**하게 됩니다. 이 부분은 논문과 다른 부분이며 필요에 따라 generator의 학습을 한번 더 진행하는 등의 시도를 해보실 수 있습니다.

또, 중간중간 이미지가 어떻게 생성되었는지를 확인하기 위해 정해진 sample_z를 이용하여 생성되는 이미지를 저장하여 비교 할 수 있도록 구현합니다. 해당 부분을 구현하게 되면 생성되는 이미지들을 가지고 다음처럼 변화 여부를 확인 할 수도 있습니다.

{% include image.html name=page.name file="GAN_samples.gif" description="GAN 학습 과정" class="large-image" %}

#### MomentumOptimizer 클래스

```python
class MomentumOptimizer(Optimizer):
    """모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스."""
    def _optimize_op(self, mode, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op.
        :param kwargs: dict, optimizer의 추가 인자.
                -momentum: float, 모멘텀 계수.
        :return tf.Operation.
        """
        
        if mode == 'generator':
            loss = self.model.gen_loss
        else:
            loss = self.model.discr_loss
        
        momentum = kwargs.pop('momentum', 0.9)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_vars = [var for var in tf.trainable_variables() if mode in var.name]
        print("{} vars will be trained for mode {}".format(len(update_vars), mode))
        for var in update_vars:
            print("{} variable has {} shape".format(var.name, var.shape))
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.AdamOptimizer(self.learning_rate_placeholder, momentum)
            .minimize(loss, var_list=update_vars)
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

`train.py` 스크립트에서 실제 학습을 수행하는 과정을 구현하며, `test.py` 스크립트에서 테스트 데이터셋에 대해 학습이 완료된 모델을 테스트하여 성능 수치를 보여주고 실제로 생성된 이미지를 그려줍니다. 또, DCGAN이 단순히 이미지를 외워서 그리는 것이 아니라 실제로 생성하는 것을 확인 하기 위해 두 이미지 사이의 interpolation 결과도 확인합니다.

### train.py 스크립트

```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('data/FFHQ/')
trainval_dir = os.path.join(root_dir, 'thumbnails128x128')

# 이미지 크기를 지정함.
IM_SIZE = (64, 64)

# 학습 셋 로드.
X_trainval = dataset.read_data(trainval_dir, IM_SIZE, 96)
trainval_size = X_trainval.shape[0]
train_set = dataset.Dataset(X_trainval)
print(train_set.num_examples)

""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정"""
hp_d = dict()

save_dir = './DCGAN_training_FFHQ_z_100/'

# FIXME: 학습 하이퍼 파라미터.
hp_d['batch_size'] = 64
hp_d['num_epochs'] = 100
hp_d['init_learning_rate'] = 2e-4
hp_d['momentum'] = 0.5
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 1.0
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
# FID를 측정하기 위한 inception 파일 경로와 미리 측정한 FFHQ의 mean, cov
hp_d['inception_path'] = 'inception/inception_v3_fr.pb'
hp_d['dataset_stats_path'] = os.path.join(trainval_dir, 'stats.pkl')
# FID를 측정할 샘플의 수
hp_d['eval_sample_size'] = 10000
hp_d['batch_size_eval'] = 50
# 학습 중간중간 이미지를 display할 설정
hp_d['sample_H'] = 20
hp_d['sample_W'] = 16
hp_d['sample_dir'] = save_dir
# 학습에 사용할 architecture parameter
hp_d['z_dim'] = 100
hp_d['G_FC_layer_channel'] = 512
hp_d['G_channel'] = 64
hp_d['D_channel'] = 64

with open(os.path.join(save_dir, 'hyperparam.json'), 'w') as f:
	json.dump(hp_d, f, indent='\t')

""" 3. Graph 생성, session 초기화 및 학습 시작 """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
model = GAN([IM_SIZE[0], IM_SIZE[1], 3], **hp_d)

evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, **hp_d)

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

	
sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, 
                                save_dir=save_dir, details=True, verbose=True, **hp_d)
```

`train.py` 스크립트에서는 마찬가지로 3단계로 진행됩니다.

1. 원본 학습 데이터셋을 메모리에 로드하고 이를 이용하여 객체 생성. 
2. 학습 관련 하이퍼파라미터 설정.
3. `ConvNet` 객체, `Evaluator` 객체 및 `Optimizer` 객체를 생성하고, TensorFlow Graph와 Session을 초기화한 뒤, `Optimizer.train` 함수를 호출하여 모델 학습을 수행함

* 원본 데이터셋 저장 경로, 하이퍼파라미터 등 `FIXME`로 표시된 부분은 여러분의 상황에 맞게 수정하시면 됩니다.

### test.py 스크립트

```python
""" 1. 원본 데이터셋을 메모리에 로드함 """
hp_d = dict()

save_dir = './DCGAN_training_FFHQ_z_100/'

with open(os.path.join(save_dir, 'hyperparam.json'), 'r') as f:
    hp_d = json.load(f)
    

# 이미지 크기를 지정함.
IM_SIZE = (64, 64)

""" 2. 테스트를 위한 하이퍼파라미터 설정 """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '1'
model = GAN([IM_SIZE[0], IM_SIZE[1], 3], **hp_d)

saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, os.path.join(save_dir, 'model_94.ckpt'))

""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
W = hp_d["sample_W"]
H = hp_d["sample_H"]

z = np.random.uniform(-1.0, 1.0, size=(W*H,hp_d["z_dim"]))

gen_img = model.generate(sess, z, verbose=True)
save_sample_images(save_dir, 'sample_random', gen_img, H, W)

""" 4. 하나의 이미지에서 다른이미지로 interpolation 수행."""
from_z = np.random.uniform(-1.0, 1.0, size=(1,hp_d["z_dim"]))
to_z = np.random.uniform(-1.0, 1.0, size=(1,hp_d["z_dim"]))

latent_intp = interpolate(from_z, to_z, 9)
img_intp = model.generate(sess, latent_intp, verbose=True)
save_sample_images(save_dir, 'interpolate', img_intp, 1, 11)

""" 5. FID 점수를 계산"""
fid = FID(hp_d["inception_path"], hp_d["dataset_stats_path"], sess)

fid.reset_FID()
sample_size = hp_d["eval_sample_size"]
sample_batch_size = hp_d['batch_size_eval']
n_batch = sample_size // sample_batch_size
for i in range(n_batch):
    eval_z = np.random.uniform(-1.0, 1.0, size=(sample_batch_size, hp_d["z_dim"]))
    g_img = model.generate(sess, eval_z)
    fid.extract_inception_features(g_img)
result = fid.calculate_FID()
print(result)
```

`test.py` 스크립트는 모델을 불러와서 여러 실험을 합니다. Random으로 이미지를 생성하기도 하고 interpolated 된 이미지를 생성하기도 하고 FID score를 계산하기도 합니다.

## 학습 결과 분석

다른 문제들과 마찬가지로 학습 수행 과정동안 학습 곡선을 그려보았습니다. 

{% include image.html name=page.name file="learning_curve-result-1.png" description="학습 곡선 플롯팅 결과<br><small>(상단 파란색: Discriminator loss, 상단 빨간색: Generator loss)(하단 파란색: 학습 batch FID, 하단 빨간색: Random하게 생성된 다수의 이미지로 측정한 FID)</small>" class="large-image" %}

학습이 진행됨에 따라, 두 Loss의 변화 양상이 다르게 되는 것을 확인 할 수 있습니다. Generator가 완전히 학습되지 않는 것을 알 수 있고 초기 단계 GAN이다 보니 완벽하게 잘 학습하는 것은 아닌 것을 확인 할 수 있습니다. FID의 경우 학습 batch의 크기가 크지 않아 일정 이상 줄어들지 않는 것을 알 수 있었고 대신 충분히 큰 크기로 sample한 데이터의 FID가 줄어드는 것으로 확인 되어 학습을 어느정도 잘 하고 있다는 것을 확인할 수 있습니다.

### 테스트 결과

가장 잘 나온 모델을 사용하여 측정한 FID의 값은 **34.73**이었습니다. 현재 state-of-the-art로 불리는 방법론들이 10이하인 것을 고려하면 아직은 부족한 결과이긴 합니다. 생성된 이미지를 보면 대부분은 잘 생성되었지만 일부 이미지가 비현실적인 것을 확인 하실 수 있습니다. 완벽하게 학습되지 않은 것을 알 수 있는 부분입니다.

{% include image.html name=page.name file="sample_image_random.jpg" description="random하게 생성된 얼굴 이미지들" class="large-image" %}

또한, 양 끝의 두 상이한 이미지에 대해 서로의 이미지로 매끄럽게 이동하는 것을 확인 하실 수 있습니다. DCGAN 논문에서 이를 **walking in the latent space**라는 재밌는 표현으로 정의하였습니다.

{% include image.html name=page.name file="sample_image_interpolate.jpg" description="안경 쓴 남자가 안경 안 쓴 여자로 바뀌는 모습" class="large-image" %}

## 결론

본 포스팅에서는 이미지를 생성하는 분야에 있어서 최근 매우 많은 주목을 받고 있는 GAN에 대하여 간략히 설명을 하였고 **Face generation**을 목표로 **DCGAN**을 Python과 Tensorflow를 이용하여 구현하였습니다. 학습에 사용한 데이터의 양이 충분하진 않았지만 어느정도 얼굴이라고 인식 할만한 이미지를 생성하는 것을 확인하였고 단순히 이미지를 기억해서 만드는 것이 아닌 이미지의 주요 feature를 학습하여 생성하는 것을 확인하였습니다. 실제 GAN을 구현해 보시는 분들에게 많은 도움이 되기를 바라는 마음에서 이 포스팅을 작성하였습니다. 도움이 되셨는지는 모르겠지만 긴 글 읽어주셔서 감사합니다. 

## References

- GAN 논문
  - <a href="http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf" target="_blank">Goodfellow et al., "Generative Adversarial Nets", NIPS, 2014</a>
- Style Transfer 관련 논문
  - <a href="https://arxiv.org/pdf/1807.10201.pdf" target="_blank">Sanakoyeu et al., "A Style-Aware Content Loss for Real-time HD Style Transfer", ECCV, 2018</a>
- DCGAN 논문
  - <a href="https://arxiv.org/abs/1511.06434" target="_blank">Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", 2016</a>
- Inception Score 관련 논문
  - <a href="https://arxiv.org/pdf/1606.03498.pdf" target="_blank">Salimans et al., "Improved Techniques for Training GANs", NIPS, 2016</a>
- Fréchet Inception Distance 관련 논문
  - <a href="https://arxiv.org/pdf/1706.08500.pdf" target="_blank">Heusel et al. "GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium", NIPS, 2017</a>
- <a href="https://github.com/NVlabs/ffhq-dataset" target="_blank">Flickr-Face-HQ Dataset</a>
- 그림
  - <a href="https://compvis.github.io/adaptive-style-transfer" target="_blank">자동차 사진을 세잔 화풍으로 전이한 이미지</a>
  - <a href="https://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html" target="_blank"> 개와 고양이를 구분하는 문제 예시</a>
  - <a href="https://commons.wikimedia.org/wiki/File:MiniDachshund1_wb.jpg" target="_blank">닥스훈트 사진</a>
  - <a href="https://www.vectorstock.com/royalty-free-vector/hand-drawn-dachshund-vector-22545919" target="_blank">닥스훈트 그림</a>
  - <a href="https://openai.com/blog/generative-models/" target="_blank">VAE 학습 과정</a>
