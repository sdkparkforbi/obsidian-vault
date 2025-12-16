

## 1. 시계열 예측의 특수성

### 1.1 일반 시퀀스 vs 시계열

```
일반 시퀀스 (텍스트):
  - 이산적 토큰
  - 순서는 중요하나 간격은 불규칙
  
시계열:
  - 연속적 값
  - 일정한 시간 간격
  - 추세, 계절성, 주기성 존재
  - 미래만 예측 (과거 참조 불가)
```

### 1.2 기존 모델의 한계

```
RNN/LSTM:
  - 순차 처리 → 느림
  - 장기 의존성 여전히 어려움

Transformer:
  - 시계열 특성 미반영
  - Positional Encoding이 시간 정보로 부족
```

---

## 2. TCN (Temporal Convolutional Network)

### 2.1 핵심 아이디어

```
CNN을 시계열에 적용
+ 인과성 보장 (미래 정보 사용 안 함)
+ 확장 합성곱으로 장기 의존성 포착
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
│             ↓    ←─────────┘ (잔차) │
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

|항목|RNN/LSTM|TCN|
|:--|:--|:--|
|구조|순환|합성곱|
|처리|순차|병렬|
|속도|느림|빠름|
|장기 의존성|LSTM: 개선|확장 합성곱으로 해결|
|메모리|많음 (상태 저장)|적음|
|기울기 흐름|복잡|직접 연결|

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
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Static  │  │  Past   │  │  Past   │  │ Future  │    │
│  │Covariates│  │Observed │  │ Known   │  │ Known   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       ↓            ↓            ↓            ↓          │
│  ┌────────────────────────────────────────────────┐    │
│  │           Variable Selection Network           │    │
│  └────────────────────────┬───────────────────────┘    │
│                           │                            │
│                           ↓                            │
│  ┌────────────────────────────────────────────────┐    │
│  │              LSTM Encoder-Decoder              │    │
│  └────────────────────────┬───────────────────────┘    │
│                           │                            │
│                           ↓                            │
│  ┌────────────────────────────────────────────────┐    │
│  │         Temporal Self-Attention Layer          │    │
│  └────────────────────────┬───────────────────────┘    │
│                           │                            │
│                           ↓                            │
│  ┌────────────────────────────────────────────────┐    │
│  │              Position-wise Feed-Forward        │    │
│  └────────────────────────┬───────────────────────┘    │
│                           │                            │
│                           ↓                            │
│                    Quantile Outputs                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
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