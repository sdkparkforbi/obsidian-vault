

## 1. 시계열 예측의 특수성

### 1.1 일반 시퀀스 vs 시계열

```
일반 시퀀스(텍스트 등):

- 이산적인 토큰(단어·서브워드·기호 등)의 나열
- 토큰의 순서는 의미에 결정적임
- 시간 개념이 없거나 약함
- 토큰 간 간격(시간·거리)은 의미를 갖지 않음
- 과거·미래 구분 없이 전체 시퀀스를 동시에 참조 가능

시계열:

- 시간에 따라 관측된 수치형 값의 흐름
- 관측 시점 간 간격이 일정함
- 시간 구조(추세·계절성·주기성)를 내포함
- 과거 관측값을 이용해 미래 값만 예측함
- 예측 시점 이후의 정보는 사용 불가(시간 인과성 제약)
```

### 1.2 기존 모델의 한계

```
NN / LSTM:

- 시점 t의 계산이 t−1에 의존하는 순차적 구조
- 시간 순서와 인과성을 구조적으로 내재함
- 병렬 처리가 어려워 시퀀스가 길수록 학습·추론 속도가 느림
- LSTM은 장기 의존성 문제를 완화했으나, 매우 긴 시계열에서는 여전히 정보 소실 가능
- 계산 효율성과 초장기 의존성 학습에 한계 존재

Transformer (self-attention 기반 구조):

- self-attention을 핵심 연산으로 사용하는 병렬 처리 아키텍처
- 모든 시점을 동시에 참조하여 장기 의존성 학습에 강점
- 기본 구조에는 시계열의 시간 간격, 추세, 계절성 개념이 내재돼 있지 않음
- Positional Encoding은 순서 정보만 제공하며 실제 시간 차이나 시간 스케일을 충분히 표현하지 못함
- 그대로 적용 시 시계열의 물리적·통계적 시간 구조를 놓칠 수 있음
    
RNN/LSTM은 “시간을 아는 대신 느린 모델”,  
Transformer는 “빠르고 전역적이지만 시간을 모르는 모델”
```

---

## 2. TCN (Temporal Convolutional Network)

### 2.1 핵심 아이디어

```
CNN을 시간축 방향으로 적용한 시계열 모델
- 공간 축 대신 시간 축에 합성곱 적용
- 모든 시점을 병렬로 처리하여 계산 효율이 높음

인과성 보장 (causal convolution)
- 시점 t의 출력은 t 및 과거 시점만 참조
- 미래 정보 사용을 구조적으로 차단하여 시계열 인과성 유지

확장 합성곱(dilated convolution)을 통한 장기 의존성 학습
- 필터 간 간격을 점진적으로 확장하여 먼 과거 정보까지 참조
- 파라미터 수 증가 없이 넓은 수용영역(receptive field) 확보
  
TCN은 순차 계산 없이 병렬 처리하면서도  
시계열의 시간 인과성과 장기 의존성을 구조적으로 보장하는 모델

예시) 

입력 문장  
나는 학교에 간다

1단계. 토큰화  
문장을 이산적 토큰 시퀀스로 분해

[나는, 학교에, 간다]

2단계. 임베딩(숫자화)  
각 토큰을 고정 길이 벡터로 변환  
(예: 임베딩 차원 = 4)

나는 → [0.2, 0.1, 0.7, 0.3]  
학교에 → [0.9, 0.4, 0.1, 0.8]  
간다 → [0.5, 0.6, 0.2, 0.1]

3단계. CNN 입력 행렬 구성  
문장 전체를 하나의 행렬로 정리  
형태: (문장 길이 × 임베딩 차원)

[ 0.2 0.1 0.7 0.3 ]  
[ 0.9 0.4 0.1 0.8 ]  
[ 0.5 0.6 0.2 0.1 ]

해석  
세로 축: 단어 순서(시간 축)  
가로 축: 단어의 의미 차원  
→ 문장은 1채널 이미지와 동일한 구조가 됨

4단계. 합성곱 적용  
필터 크기 = 2인 경우

첫 번째 합성곱 영역  
[나는, 학교에]

두 번째 합성곱 영역  
[학교에, 간다]

각 영역에서 연속된 두 단어의 의미 패턴을 추출

5단계. 의미 해석  
CNN은  
“나는 → 학교에”  
“학교에 → 간다”  
와 같은 국소적인 n-그램 의미 패턴을 병렬로 학습

```

### 2.2 구성 요소

**1) 인과적 합성곱 (Causal Convolution)**

```
일반 합성곱:
  과거 + 현재 + 미래 → 출력

인과적 합성곱:
  과거 + 현재만 → 출력 (미래 사용 안 함)

시점 t의 출력:
  yₜ = f(xₜ, xₜ₋₁, xₜ₋₂, ...)
      (xₜ₊₁, xₜ₊₂ 사용 안 함!)
```

**시각화:**

```
        y₁    y₂    y₃    y₄    y₅
         ↑     ↑     ↑     ↑     ↑
        /|\   /|\   /|\   /|\   /|\
       / | \ / | \ / | \ / | \ / | \
      x₁ x₂ x₃ x₄ x₅

→ y₃은 x₁, x₂, x₃만 참조 (x₄, x₅는 안 봄)
```

**2) 확장 합성곱 (Dilated Convolution)**

```
문제: 일반 합성곱은 수용 영역(receptive field)이 작음
해결: 확장률(dilation)을 점점 키움

Layer 1: dilation = 1
  ──○──○──○──
    │  │  │
    x₁ x₂ x₃

Layer 2: dilation = 2
  ──○─────○─────○──
    │     │     │
    x₁    x₃    x₅

Layer 3: dilation = 4
  ──○─────────○─────────○──
    │         │         │
    x₁        x₅        x₉

→ 적은 층으로 긴 시퀀스 포착 가능!
```

**3) 잔차 연결 (Residual Connection)**

```
출력 = 입력 + Conv(입력)

→ 깊은 네트워크에서 기울기 소실 방지
→ 학습 안정화
```

### 2.3 TCN 블록

```
┌─────────────────────────────────────┐
│                                     │
│  Input                              │
│    │                                │
│    ├───────────────────────┐        │
│    ↓                       │        │
│  ┌─────────────────────┐   │        │
│  │ Dilated Causal Conv │   │        │
│  └──────────┬──────────┘   │        │
│             ↓              │        │
│  ┌─────────────────────┐   │        │
│  │      ReLU           │   │        │
│  └──────────┬──────────┘   │        │
│             ↓              │        │
│  ┌─────────────────────┐   │        │
│  │     Dropout         │   │        │
│  └──────────┬──────────┘   │        │
│             ↓              │        │
│  ┌─────────────────────┐   │        │
│  │ Dilated Causal Conv │   │        │
│  └──────────┬──────────┘   │        │
│             ↓              │        │
│  ┌─────────────────────┐   │        │
│  │      ReLU           │   │        │
│  └──────────┬──────────┘   │        │
│             ↓              │        │
│  ┌─────────────────────┐   │        │
│  │     Dropout         │   │        │
│  └──────────┬──────────┘   │        │
│             │              │        │
│             ↓    ←─────────┘ (잔차)  │
│           (+)                       │
│             ↓                       │
│          Output                     │
│                                     │
└─────────────────────────────────────┘
```

### 2.4 수용 영역 계산

```
Receptive Field = 1 + (kernel_size - 1) × Σᵢ dilation_rate[i]

예: kernel=3, dilations=[1,2,4,8]
RF = 1 + (3-1) × (1+2+4+8)
   = 1 + 2 × 15
   = 31

→ 4개 층으로 31 시점 커버!
```

### 2.5 Python 코드 (PyTorch)

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # 미래 부분 제거

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 잔차 연결 (채널 수가 다르면 1x1 conv)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
                        if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return self.relu(out + self.residual(x))

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]  # 마지막 시점
        return self.fc(out)

# 사용 예시
model = TCN(input_size=1, output_size=1, num_channels=[32, 32, 32, 32])
x = torch.randn(16, 100, 1)  # batch=16, seq_len=100, features=1
y = model(x)  # (16, 1)
```

### 2.6 TCN vs RNN/LSTM

| 항목     | RNN/LSTM   | TCN         |
| :----- | :--------- | :---------- |
| 구조     | 순환         | 합성곱         |
| 처리     | 순차         | 병렬          |
| 속도     | 느림         | 빠름          |
| 장기 의존성 | LSTM: 개선   | 확장 합성곱으로 해결 |
| 메모리    | 많음 (상태 저장) | 적음          |
| 기울기 흐름 | 복잡         | 직접 연결       |

---

## 3. TFT (Temporal Fusion Transformer)

### 3.1 핵심 아이디어

```
시계열 예측을 위한 Transformer
+ 정적/동적 변수 처리
+ 과거/미래 알려진 정보 구분
+ 해석 가능한 Attention
```

### 3.2 시계열 예측의 입력 유형

```
1) 정적 변수 (Static Covariates)
   - 시간에 따라 변하지 않음
   - 예: 매장 ID, 지역, 카테고리

2) 과거 관측값 (Past Observed)
   - 과거에만 알 수 있음
   - 예: 과거 매출, 과거 방문자 수

3) 과거 알려진 입력 (Past Known)
   - 과거 시점에서 알 수 있었던 정보
   - 예: 과거 날짜, 과거 요일

4) 미래 알려진 입력 (Future Known)
   - 미래 시점에도 알 수 있는 정보
   - 예: 미래 날짜, 공휴일, 예정된 이벤트
```

### 3.3 TFT 전체 구조

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐    │
│  │  Static   │  │   Past   │  │   Past    │  │  Future   │    │
│  │ Covariates│  │ Observed │  │   Known   │  │  Known    │    │
│  └─────┬─────┘  └─────┬────┘  └─────┬─────┘  └─────┬─────┘    │
│        │              │             │              │          │
│        ↓              ↓             ↓              ↓          │
│  ┌───────────────────────────────────────────────────────┐    │
│  │               Variable Selection Network              │    │
│  └────────────────────────────┬──────────────────────────┘    │
│                               │                               │
│                               ↓                               │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                   LSTM Encoder-Decoder                │    │
│  └────────────────────────────┬──────────────────────────┘    │
│                               │                               │
│                               ↓                               │
│  ┌───────────────────────────────────────────────────────┐    │
│  │               Temporal Self-Attention Layer           │    │
│  └────────────────────────────┬──────────────────────────┘    │
│                               │                               │
│                               ↓                               │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                  Position-wise Feed-Forward           │    │
│  └────────────────────────────┬──────────────────────────┘    │
│                               │                               │
│                               ↓                               │
│                        Quantile Outputs                       │
│                                                               │
└───────────────────────────────────────────────────────────────┘

TFT는  
LSTM 기반 시간 요인  
attention 기반 관계 요인  
변수 선택 기반 supervised factor estimation  
을 하나로 융합한 모델

그래서 “Fusion”이라는 이름이 붙음

```

### 3.4 주요 컴포넌트

**1) Variable Selection Network**

```
"어떤 변수가 중요한가?"

각 변수의 중요도 가중치 학습
→ 해석 가능성 향상
→ 불필요한 변수 자동 제외
```

**2) Gated Residual Network (GRN)**

```
GRN(x, c) = LayerNorm(x + GLU(η₁))
η₁ = W₁·η₂ + b₁
η₂ = ELU(W₂·x + W₃·c + b₂)

GLU = Gated Linear Unit
    = σ(W₄·η₁ + b₄) ⊙ (W₅·η₁ + b₅)

→ 필요한 정보만 통과시킴
```

**3) Interpretable Multi-Head Attention**

```
일반 Attention: 여러 head 결합 후 해석 어려움

TFT Attention:
  - 각 head의 가중치 공유
  - Attention 가중치 직접 해석 가능
  
  "언제가 중요했는가?" 시각화 가능
```

**4) Quantile Output**

```
점 예측 대신 분위수 예측

출력: q10, q50, q90
    = 10%, 50%, 90% 분위수

→ 예측 불확실성 정량화
→ 의사결정에 활용
```

### 3.5 Python 코드 (PyTorch Forecasting)

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

# 데이터셋 생성
training = TimeSeriesDataSet(
    data=df,
    time_idx="time_idx",
    target="target",
    group_ids=["group"],
    min_encoder_length=24,
    max_encoder_length=48,
    min_prediction_length=1,
    max_prediction_length=12,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx", "month", "day_of_week"],
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["group"]),
)

# 모델 생성
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)

# 학습
trainer = pl.Trainer(max_epochs=30, gpus=1)
trainer.fit(tft, train_dataloaders=train_dataloader)

# 예측
predictions = tft.predict(val_dataloader)

# 해석
interpretation = tft.interpret_output(predictions)
tft.plot_interpretation(interpretation)
```

### 3.6 TFT의 장점

```
1. 다양한 입력 처리
   - 정적/동적, 과거/미래 구분

2. 해석 가능성
   - 변수 중요도
   - Attention 시각화

3. 불확실성 정량화
   - Quantile 예측

4. 장기 예측
   - Multi-horizon 지원
```

---

## 4. TCN vs TFT 비교

|항목|TCN|TFT|
|:--|:--|:--|
|기반|CNN|Transformer + LSTM|
|입력 유형|단순 시계열|다양한 변수|
|해석 가능성|낮음|높음|
|불확실성|없음 (기본)|Quantile|
|계산 비용|낮음|높음|
|적합한 경우|단변량, 빠른 처리|다변량, 해석 필요|

---

## 5. 실제 적용 예시

### 5.1 매출 예측 (TFT)

```
정적 변수: 매장 ID, 지역, 매장 크기
과거 관측값: 과거 매출
과거 알려진: 과거 날짜, 요일, 월
미래 알려진: 미래 날짜, 공휴일, 프로모션 일정

출력: 향후 7일 매출 (10%, 50%, 90% 분위수)
```

### 5.2 전력 수요 예측 (TCN)

```
입력: 과거 24시간 전력 사용량
출력: 다음 1시간 전력 수요

TCN 선택 이유:
- 단변량 시계열
- 빠른 추론 필요 (실시간)
- 높은 샘플링 빈도
```

### 5.3 약동학 예측 (TFT 응용)

```
정적 변수: 환자 ID, 성별, 나이, 체중
과거 관측값: 과거 혈중 농도
과거 알려진: 투약 시간, 용량
미래 알려진: 예정된 투약 일정

출력: 향후 혈중 농도 곡선 + 불확실성
```

---

## 6. 요약

### 6.1 TCN

```
인과적 합성곱 + 확장 합성곱
병렬 처리 → 빠름
긴 시퀀스 효율적 처리
단순한 시계열에 적합
```

### 6.2 TFT

```
Transformer + LSTM + Variable Selection
다양한 입력 유형 처리
해석 가능한 Attention
Quantile 예측으로 불확실성
복잡한 예측 문제에 적합
```

### 6.3 선택 가이드

```
단변량, 빠른 처리 → TCN
다변량, 해석 필요 → TFT
장기 의존성 중요 → 둘 다 가능
불확실성 필요 → TFT
```

---

## 참고문헌

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. _arXiv preprint arXiv:1803.01271_.
    
2. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. _International Journal of Forecasting_, 37(4), 1748-1764.
    
3. Oord, A. V. D., et al. (2016). WaveNet: A generative model for raw audio. _arXiv preprint arXiv:1609.03499_.