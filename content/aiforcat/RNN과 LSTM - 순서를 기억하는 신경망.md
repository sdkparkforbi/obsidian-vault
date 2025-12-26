
## 1. Word2Vec의 한계

### 1.1 복습: Word2Vec

```
입력: 단어
출력: 단어 벡터 (고정)

"사과" → [0.8, 0.2, 0.1] (항상 같은 벡터)
```

### 1.2 문제: 순서 정보 손실

```
문장 1: "개가 사람을 물었다"
문장 2: "사람이 개를 물었다"

Word2Vec 방식:
  Bag of Words = {개, 사람, 물었다}
  → 두 문장이 같은 표현!
  → 의미 차이 반영 불가
```

### 1.3 필요한 것

```
순서를 고려하는 모형

"개가" → "사람을" → "물었다"
   ↓         ↓          ↓
  상태1 → 상태2 → 상태3 → 출력
```

---

## 2. RNN (Recurrent Neural Network)

### 2.1 핵심 아이디어

**이전 상태를 기억하면서 다음 입력 처리**

```
시점 1: x₁ → h₁
시점 2: x₂ + h₁ → h₂
시점 3: x₃ + h₂ → h₃
...

h = 은닉 상태 (hidden state)
  = 지금까지 읽은 내용의 "요약"
```

### 2.2 모형 구조

```
      x₁        x₂        x₃        x₄
      ↓         ↓         ↓         ↓
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
h₀→ │ A │ →h₁→│ A │ →h₂→│ A │ →h₃→│ A │ →h₄
    └───┘     └───┘     └───┘     └───┘
      ↓         ↓         ↓         ↓
      y₁        y₂        y₃        y₄

A = 같은 신경망 (가중치 공유)
```

### 2.3 수식

**은닉 상태 업데이트:**

```
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
```

**출력 계산:**

```
yₜ = Wₕᵧ · hₜ + bᵧ
```

**모수:**

- Wₓₕ: 입력 → 은닉 (d_in × d_h)
- Wₕₕ: 은닉 → 은닉 (d_h × d_h)
- Wₕᵧ: 은닉 → 출력 (d_h × d_out)
- bₕ, bᵧ: 편향

### 2.4 간단한 예시

**설정:**

```
어휘: [나는, 사과를, 먹었다] (V=3)
임베딩 차원: d_in = 2
은닉 차원: d_h = 2
```

**입력 문장:** "나는 사과를 먹었다"

**시점 1: "나는"**

```
x₁ = [0.1, 0.9]  (나는의 임베딩)
h₀ = [0, 0]      (초기 상태)

h₁ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀)
   = tanh([0.5, 0.3])
   = [0.46, 0.29]
```

**시점 2: "사과를"**

```
x₂ = [0.8, 0.2]  (사과를의 임베딩)
h₁ = [0.46, 0.29]

h₂ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁)
   = tanh([0.7, 0.5])
   = [0.60, 0.46]
```

**시점 3: "먹었다"**

```
x₃ = [0.3, 0.7]  (먹었다의 임베딩)
h₂ = [0.60, 0.46]

h₃ = tanh(Wₓₕ·x₃ + Wₕₕ·h₂)
   = [0.71, 0.58]

→ h₃ = 문장 전체의 벡터 표현!
```

---

## 3. RNN의 문제: Vanishing Gradient

### 3.1 문제 상황

**긴 문장:**

```
"어제 비가 와서 우산을 가져갔는데 오늘은 맑아서 ___ 를 두고 왔다"

정답: "우산"
필요한 정보: 문장 초반의 "우산"
```

### 3.2 기울기 소실

```
역전파 시:

∂L/∂h₁ = ∂L/∂h₂ × ∂h₂/∂h₁
       = ∂L/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁
       ...

∂hₜ/∂hₜ₋₁ = Wₕₕ × tanh'(...)

tanh'(x) ∈ (0, 1]
Wₕₕ의 원소 < 1이면

→ 곱이 반복될수록 기울기 → 0
→ 먼 과거 정보 학습 불가
```

### 3.3 시각화

```
기울기 크기

  │
1 │████
  │███
  │██
  │█
  │▪
0 └──────────────→ 시점
     1  2  3  4  5

→ 시점 1의 정보가 시점 5에 거의 전달 안 됨
```

---

## 4. LSTM (Long Short-Term Memory)

### 4.1 핵심 아이디어

**게이트로 정보 흐름 조절**

```
RNN: 모든 정보를 무조건 섞음
LSTM: 어떤 정보를 기억/삭제/출력할지 선택
```

### 4.2 LSTM 구조

```
         ┌─────────────────────────────────┐
         │           Cell State (C)        │
         │    ───────────────────────→     │
         │      ↑         ↑         ↓      │
         │   forget    input    output     │
         │    gate      gate     gate      │
         │      ↑         ↑         ↓      │
         │    ┌───┐     ┌───┐    ┌───┐     │
hₜ₋₁ ──→  ├──→ │ f │     │ i │    │ o │──→─ ┤──→ hₜ
         │    └───┘     └───┘    └───┘     │
         │                                 │
xₜ ────→  ├─────────────────────────────────┤
         └─────────────────────────────────┘
```

### 4.3 세 가지 게이트

**1) Forget Gate (삭제 게이트)**

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

"이전 기억 중 얼마나 잊을까?"
fₜ ∈ (0, 1)
0 = 완전히 잊음
1 = 완전히 기억
```

**2) Input Gate (입력 게이트)**

```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

"새 정보 중 얼마나 저장할까?"
iₜ = 저장 비율
C̃ₜ = 저장할 후보 정보
```

**3) Output Gate (출력 게이트)**

```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

"기억 중 얼마나 출력할까?"
```

### 4.4 Cell State 업데이트

```
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
     ─────────   ─────────
     이전 기억    새 정보
     (일부 삭제)  (일부 추가)
```

### 4.5 Hidden State 출력

```
hₜ = oₜ × tanh(Cₜ)
```

---

## 5. LSTM 예시

### 5.1 설정

```
문장: "비가 와서 우산을 가져갔다"
토큰: [비가, 와서, 우산을, 가져갔다]

관심: "우산" 정보가 끝까지 유지되는가?
```

### 5.2 시점별 처리

**시점 1: "비가"**

```
입력: x₁ = "비가" 임베딩
초기: h₀ = 0, C₀ = 0

Forget Gate: f₁ = 0.1 (초기, 잊을 것 없음)
Input Gate:  i₁ = 0.8 (새 정보 많이 저장)
새 정보:     C̃₁ = "비" 관련 정보

Cell State:  C₁ = 0.1×0 + 0.8×C̃₁ = "비 정보"
```

**시점 2: "와서"**

```
입력: x₂ = "와서" 임베딩

Forget Gate: f₂ = 0.9 (이전 정보 유지)
Input Gate:  i₂ = 0.3 (보조 정보만 추가)

Cell State:  C₂ = 0.9×C₁ + 0.3×C̃₂
            = "비 정보" (유지) + "동작 정보" (추가)
```

**시점 3: "우산을"**

```
입력: x₃ = "우산을" 임베딩

Forget Gate: f₃ = 0.7 (일부 유지)
Input Gate:  i₃ = 0.9 (중요 정보! 많이 저장)

Cell State:  C₃ = 0.7×C₂ + 0.9×C̃₃
            = "비 + 우산 정보"
```

**시점 4: "가져갔다"**

```
입력: x₄ = "가져갔다" 임베딩

Forget Gate: f₄ = 0.8 (우산 정보 유지!)
Input Gate:  i₄ = 0.5

Cell State:  C₄ = 0.8×C₃ + 0.5×C̃₄
            = "우산 정보" (여전히 남아있음!)

Output Gate: o₄ = 0.9
Hidden:      h₄ = 0.9 × tanh(C₄)
            → "우산을 가져갔다" 의미 포함
```

### 5.3 핵심

```
Cell State가 "고속도로" 역할

정보가 변형 없이 (또는 최소 변형으로) 전달
→ 먼 과거 정보도 유지 가능
→ Vanishing Gradient 완화
```

---

## 6. RNN vs LSTM 비교

|항목|RNN|LSTM|
|:--|:--|:--|
|구조|단순|게이트 3개 + Cell State|
|모수|적음|많음 (약 4배)|
|장기 기억|어려움|가능|
|학습 속도|빠름|느림|
|Vanishing Gradient|심각|완화|

---

## 7. GRU (Gated Recurrent Unit)

### 7.1 LSTM의 간소화 버전

```
LSTM: 게이트 3개 (forget, input, output)
GRU:  게이트 2개 (reset, update)
```

### 7.2 GRU 수식

```
Reset Gate:  rₜ = σ(Wr · [hₜ₋₁, xₜ])
Update Gate: zₜ = σ(Wz · [hₜ₋₁, xₜ])

후보:        h̃ₜ = tanh(W · [rₜ × hₜ₋₁, xₜ])
출력:        hₜ = (1-zₜ) × hₜ₋₁ + zₜ × h̃ₜ
```

### 7.3 LSTM vs GRU

|항목|LSTM|GRU|
|:--|:--|:--|
|게이트 수|3개|2개|
|Cell State|별도 존재|없음 (h만 사용)|
|모수|더 많음|더 적음|
|성능|대체로 비슷|대체로 비슷|
|학습 속도|느림|빠름|

---

## 8. 시계열 예측에 적용

### 8.1 문제 설정

```
시계열: [y₁, y₂, y₃, ..., yₜ]
목표: yₜ₊₁ 예측
```

### 8.2 RNN/LSTM 구조

```
y₁    y₂    y₃    ...    yₜ
 ↓     ↓     ↓           ↓
┌─┐   ┌─┐   ┌─┐         ┌─┐
│ │──→│ │──→│ │──→...──→│ │──→ ŷₜ₊₁
└─┘   └─┘   └─┘         └─┘
```

### 8.3 Python 코드 (PyTorch)

https://colab.research.google.com/drive/1-ET8pfcsVvQ4sxPR2VwrJNeGEWcCpZBC?usp=sharing

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (1, batch, hidden_size)
        out = self.fc(h_n.squeeze(0))
        return out

# 모델 생성
model = LSTMPredictor(input_size=1, hidden_size=32, output_size=1)

# 예시 입력
x = torch.randn(16, 10, 1)  # batch=16, seq_len=10, features=1
y_pred = model(x)  # (16, 1)
```

---

## 9. 텍스트 분류에 적용

### 9.1 문제 설정

```
입력: 문장 (단어 시퀀스)
출력: 클래스 (긍정/부정, 주제 등)
```

### 9.2 구조

```
"이 영화 정말 재미있다"

[이]   [영화]  [정말]  [재미있다]
  ↓      ↓       ↓        ↓
┌───┐  ┌───┐  ┌───┐    ┌───┐
│   │─→│   │─→│   │─→  │   │─→ h_final
└───┘  └───┘  └───┘    └───┘
                          ↓
                    ┌─────────┐
                    │ Softmax │
                    └─────────┘
                          ↓
                    [긍정: 0.9]
                    [부정: 0.1]
```

### 9.3 Python 코드

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) - 단어 인덱스
        embeds = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        out = self.fc(h_n.squeeze(0))
        return out
```

---

## 10. 요약

### 10.1 RNN

```
이전 상태 + 현재 입력 → 현재 상태
순서 정보 반영 가능
문제: Vanishing Gradient (장기 기억 어려움)
```

### 10.2 LSTM

```
게이트로 정보 흐름 조절
Cell State = 장기 기억 저장소
Forget, Input, Output Gate
→ Vanishing Gradient 완화
```

### 10.3 GRU

```
LSTM의 간소화 버전
게이트 2개 (Reset, Update)
성능 비슷, 계산 빠름
```

### 10.4 한계

```
입력 → 은닉 → 출력

문제: 입력과 출력의 길이가 다르면?
예: 번역 "나는 학생이다" → "I am a student"

→ 해결: Seq2Seq (다음 강의)
```

---

## 참고문헌

1. Elman, J. L. (1990). Finding structure in time. _Cognitive science_, 14(2), 179-211.
    
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural computation_, 9(8), 1735-1780.
    
3. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_.