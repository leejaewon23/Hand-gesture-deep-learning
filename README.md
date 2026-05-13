# 👏 Hand Gesture Deep Learning

PyTorch와 Google Colab을 이용해 LeapGestRecog 손 제스처 이미지를 분류하는 딥러닝 실험 저장소입니다. 기본 CNN 모델부터 EfficientNet-B3 전이학습 모델까지 비교하며, 최종 노트북에는 Gradio 기반 예측 데모까지 포함되어 있습니다.

## 프로젝트 개요

이 프로젝트는 Kaggle의 LeapGestRecog 데이터셋을 사용해 10개 손 제스처 클래스를 분류합니다. 데이터 누수를 줄이기 위해 이미지 단위 무작위 분할이 아니라 피험자 ID 기준으로 학습, 검증, 테스트 세트를 나눕니다.

- 학습 데이터: 피험자 `00`~`05`
- 검증 데이터: 피험자 `06`~`07`
- 테스트 데이터: 피험자 `08`~`09`
- 클래스 수: 10개
- 주요 모델: 기본 CNN, 개선 CNN, EfficientNet-B3 전이학습
- 실행 환경: Google Colab GPU

## 저장소 구성

| 파일 | 설명 |
| --- | --- |
| `손_제스처_인식_모델_설계(중간).ipynb` | EfficientNet-B3 기반 최종 손 제스처 인식 노트북입니다. 데이터 탐색, 전처리, 학습, 평가, Gradio 데모를 포함합니다. |
| `hand_gesture_cnn_simple_colab.ipynb` | Colab에서 바로 실행할 수 있는 단순 CNN 손 제스처 분류 노트북입니다. |
| `hand_gesture_cnn_baseline.ipynb` | CIFAR 실험에서 사용한 CNN 구조를 손 제스처 10클래스 문제에 맞춘 기준 모델 노트북입니다. |
| `CNN.ipynb` | CIFAR-100으로 CNN 구조, 필터, 특징 맵, 혼동 행렬을 실험한 참고 노트북입니다. |
| `자연_장면_분류.ipynb` | Intel Image Classification 데이터셋으로 자연 장면 분류를 실습한 노트북입니다. |
| `자연_사진_분류_2차_ipynb의_사본.ipynb` | Intel Image Classification 데이터셋에 timm 모델, Albumentations, MixUp/CutMix, TTA 등을 적용한 고도화 실험 노트북입니다. |
| `.gitignore` | Python 캐시 파일 제외 설정입니다. |

## 주요 기능

- Kaggle API를 통한 데이터셋 다운로드 및 압축 해제
- 피험자 독립 train/validation/test 분할
- PyTorch `Dataset`과 `DataLoader` 기반 이미지 로딩
- 기본 CNN 및 EfficientNet-B3 전이학습 모델 학습
- 이미지 증강, 정규화, label smoothing, MixUp, RandAugment 적용
- 정확도, classification report, confusion matrix 기반 평가
- 학습 곡선 및 예측 샘플 시각화
- Gradio를 이용한 이미지 업로드/웹캠 예측 데모

## 데이터셋

### LeapGestRecog

손 제스처 인식 노트북은 Kaggle의 `gti-upm/leapgestrecog` 데이터셋을 사용합니다.

노트북에서 사용하는 클래스는 다음과 같습니다.

```text
01_palm
02_l
03_fist
04_fist_moved
05_thumb
06_index
07_ok
08_palm_moved
09_c
10_down
```

최종 노트북 기준 데이터 분할 수는 다음과 같습니다.

| Split | 피험자 | 이미지 수 |
| --- | --- | ---: |
| Train | `00`, `01`, `02`, `03`, `04`, `05` | 12,000 |
| Validation | `06`, `07` | 4,000 |
| Test | `08`, `09` | 4,000 |

### 보조 실험 데이터셋

- `CNN.ipynb`: `torchvision.datasets.CIFAR100`
- `자연_장면_분류.ipynb`, `자연_사진_분류_2차_ipynb의_사본.ipynb`: Kaggle `puneet6060/intel-image-classification`

## 실행 환경

권장 환경은 Google Colab GPU 런타임입니다.

- Python 3
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- Pillow
- kaggle
- gradio

고도화된 자연 이미지 분류 노트북에서는 추가로 다음 패키지를 사용합니다.

- albumentations
- timm
- torchmetrics
- opencv-python
- pandas
- tqdm

## 빠른 시작

1. 이 저장소를 클론합니다.

```bash
git clone https://github.com/leejaewon23/Hand-gesture-deep-learning.git
cd Hand-gesture-deep-learning
```

2. Google Colab에서 실행할 노트북을 엽니다.

권장 실행 순서는 다음과 같습니다.

```text
1. hand_gesture_cnn_simple_colab.ipynb
2. hand_gesture_cnn_baseline.ipynb
3. 손_제스처_인식_모델_설계(중간).ipynb
```

3. Colab 런타임을 GPU로 변경합니다.

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

4. Kaggle 계정에서 `kaggle.json` API 토큰을 내려받습니다.

```text
Kaggle > Account > API > Create New Token
```

5. 노트북의 `kaggle.json` 업로드 셀을 실행하고, 이후 셀을 순서대로 실행합니다.

## 모델 요약

### Simple CNN

`hand_gesture_cnn_simple_colab.ipynb`는 작은 CNN 모델로 빠르게 전체 학습 과정을 확인하는 노트북입니다.

- 입력 크기: `128 x 128`
- Optimizer: Adam
- Epochs: 20
- 저장 파일: `/content/best_hand_gesture_cnn.pt`
- 최고 검증 정확도: 93.35%
- 테스트 정확도: 75.95%

### CNN Baseline

`hand_gesture_cnn_baseline.ipynb`는 더 깊은 CNN 블록 구조를 사용합니다.

- 입력 크기: `128 x 128`
- Conv-BatchNorm-ReLU 블록 기반 구조
- Optimizer: SGD + momentum + Nesterov
- Scheduler: CosineAnnealingLR
- Epochs: 50
- 출력 폴더: `./cnn_results`

### EfficientNet-B3

`손_제스처_인식_모델_설계(중간).ipynb`는 ImageNet 사전학습 EfficientNet-B3를 손 제스처 분류에 맞게 fine-tuning합니다.

- 입력 크기: `300 x 300`
- Batch size: 16
- 1단계: backbone freeze 후 classifier warm-up
- 2단계: 전체 레이어 fine-tuning
- 주요 기법: RandAugment, RandomErasing, MixUp, label smoothing, gradient clipping, cosine warm restart
- 저장 파일: `/content/gesture_efficientnet_b3.pth`
- 최고 검증 정확도: 99.55%
- 테스트 정확도: 약 97%
- Gradio 데모 포함

## 결과 예시

EfficientNet-B3 최종 테스트 결과는 피험자 `08`, `09` 기준으로 약 97%의 정확도를 보였습니다. 단순 CNN은 검증 정확도는 높게 나왔지만 테스트 정확도는 75.95%로 낮아, 피험자 독립 테스트에서 일반화 성능 차이가 크게 나타났습니다.

## 출력 파일

노트북 실행 시 다음과 같은 결과물이 생성됩니다.

| 경로 | 설명 |
| --- | --- |
| `/content/best_hand_gesture_cnn.pt` | Simple CNN 최고 검증 성능 모델 가중치 |
| `./cnn_results/best_cnn_baseline.pt` | CNN baseline 모델 가중치 |
| `./cnn_results/cnn_confusion_matrix.png` | CNN baseline 혼동 행렬 |
| `./cnn_results/cnn_learning_curve.png` | CNN baseline 학습 곡선 |
| `/content/gesture_efficientnet_b3.pth` | EfficientNet-B3 최고 검증 성능 모델 가중치 |

## 참고 노트북 활용 방법

손 제스처 모델을 개선하고 싶다면 다음 순서로 참고하면 좋습니다.

1. `CNN.ipynb`에서 기본 CNN 구조와 시각화 방법을 확인합니다.
2. `hand_gesture_cnn_simple_colab.ipynb`에서 손 제스처 데이터셋 로딩과 기본 학습 흐름을 확인합니다.
3. `hand_gesture_cnn_baseline.ipynb`에서 더 깊은 CNN 구조와 평가 저장 방식을 확인합니다.
4. `손_제스처_인식_모델_설계(중간).ipynb`에서 전이학습, 증강, Gradio 데모를 적용합니다.
5. 자연 이미지 분류 노트북에서 timm, Albumentations, TTA 같은 고도화 기법을 참고합니다.

## 주의사항

- Kaggle 데이터셋은 저장소에 포함되어 있지 않습니다. 각 노트북에서 Kaggle API로 직접 다운로드해야 합니다.
- `kaggle.json`은 개인 인증 파일이므로 저장소에 커밋하지 않아야 합니다.
- 대부분의 노트북은 Colab 경로(`/content/...`)를 기준으로 작성되어 있습니다.
- EfficientNet-B3 학습은 GPU 메모리를 많이 사용하므로, 메모리가 부족하면 batch size를 줄이세요.
- 현재 저장소에는 별도 라이선스 파일이 없습니다. 공개 배포나 재사용을 고려한다면 `LICENSE` 파일을 추가하는 것이 좋습니다.

## 향후 개선 방향

- `requirements.txt` 또는 `environment.yml` 추가
- 모델별 결과를 표준화해 비교표 작성
- 학습된 가중치와 추론 스크립트 분리
- Gradio 데모를 독립 실행 가능한 Python 파일로 정리
- README에 결과 이미지와 혼동 행렬 예시 추가
