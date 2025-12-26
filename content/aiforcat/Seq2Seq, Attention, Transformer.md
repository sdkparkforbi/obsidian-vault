
## 1. RNN/LSTM의 한계

### 1.1 복습

```
RNN/LSTM:
  입력 시퀀스 → 은닉 상태 → 출력

"나는 학생이다" → h₄ → ???

여기서, 출력은 감성분류, 다음 단어 예측 혹은 h₄ 자체를 벡터로 사용  
```

### 1.2 문제: 입출력 길이가 다를 때

```
번역:
  입력: "나는 학생이다" (3 토큰)
  출력: "I am a student" (4 토큰)

→ 입력과 출력의 길이가 다름!
→ 단순 RNN으로는 처리 어려움

RNN, LSTM, GRU는 분류(클래스 2개)든 회귀(숫자 1개)든 미리 정해진 크기만 출력
번역, 요약, 질의응답처럼 출력 길이가 가변인 문제는 Encoder-Decoder 구조가 필요

```

---

## 2. Seq2Seq (Sequence to Sequence)

### 2.1 핵심 아이디어

```
두 개의 RNN:
  Encoder: 입력 → 고정 벡터 (Context Vector)
  Decoder: 고정 벡터 → 출력
```

| 모델          | Encoder             | Decoder             |
| ----------- | ------------------- | ------------------- |
| 초기 Seq2Seq  | RNN/LSTM            | RNN/LSTM            |
| Transformer | Self-Attention      | Self-Attention      |
| BERT        | Transformer Encoder | -                   |
| GPT         | -                   | Transformer Decoder |

### 2.2 구조

```
        Encoder                    Decoder
        
"나는" "학생" "이다"           "I"  "am"  "a"  "student"
   ↓      ↓      ↓              ↓    ↓     ↓      ↓
 ┌───┐  ┌───┐  ┌───┐        ┌───┐ ┌───┐ ┌───┐  ┌───┐
 │   │→ │   │→ │   │→ [c] → │   │→│   │→│   │→ │   │
 └───┘  └───┘  └───┘        └───┘ └───┘ └───┘  └───┘
                                ↓    ↓     ↓      ↓
                              "am"  "a" "student" <EOS>
```

### 2.3 Encoder

```
입력: x₁, x₂, ..., xₙ (원문)
출력: Context Vector c

h₁ = RNN(x₁, h₀)
h₂ = RNN(x₂, h₁)
...
hₙ = RNN(xₙ, hₙ₋₁)

c = hₙ  (마지막 은닉 상태)
```

### 2.4 Decoder

```
입력: Context Vector c, 이전 출력 yₜ₋₁
출력: 다음 단어 yₜ

s₁ = RNN(c, <SOS>)     → y₁
s₂ = RNN(s₁, y₁)       → y₂
...
sₘ = RNN(sₘ₋₁, yₘ₋₁)   → yₘ = <EOS>
```

### 2.5 문제점: 정보 병목

```
모든 입력 정보 → 하나의 벡터 c

긴 문장:
"어제 비가 많이 와서 우산을 가져갔는데..."
              ↓
          c = [0.3, 0.5, ...]  (고정 크기!)
              ↓
"Yesterday it rained a lot so I brought..."

→ 정보 손실 불가피
→ 특히 긴 문장에서 성능 저하
```

---

## 3. Attention Mechanism

### 3.1 핵심 아이디어

```
번역할 때마다 "원문의 어디를 볼까?" 결정

"I" 생성 시 → "나는"에 집중
"student" 생성 시 → "학생"에 집중
```

### 3.2 구조

```
       Encoder 은닉 상태들
         h₁    h₂    h₃
          \    |    /
           \   |   /
        Attention Weights
           α₁  α₂  α₃
            \  |  /
             \ | /
               ↓
        Context Vector (가중 평균)
               ↓
            Decoder
```

### 3.3 Attention 계산

**Step 1: 유사도 점수 (Score)**

```
eᵢⱼ = score(sⱼ₋₁, hᵢ)

sⱼ₋₁ = Decoder의 이전 상태
hᵢ   = Encoder의 i번째 상태

score 함수 종류:
- Dot:      sⱼ₋₁ · hᵢ
- General:  sⱼ₋₁ · W · hᵢ
- Concat:   v · tanh(W[sⱼ₋₁; hᵢ])
```

**Step 2: Attention 가중치 (Softmax)**

```
αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eₖⱼ)

→ 모든 가중치 합 = 1
→ 어디에 집중할지 확률로 표현
```

**Step 3: Context Vector (가중 평균)**

```
cⱼ = Σᵢ αᵢⱼ · hᵢ
```

**Step 4: Decoder 출력**

```
sⱼ = RNN(sⱼ₋₁, [yⱼ₋₁; cⱼ])
yⱼ = softmax(W · sⱼ)
```

### 3.4 예시: 번역

```
원문: "나는 학생이다"
번역: "I am a student"

Encoder 은닉 상태:
  h₁ = "나는" 정보
  h₂ = "학생" 정보
  h₃ = "이다" 정보

"I" 생성 시:
  α = [0.8, 0.1, 0.1]  → "나는"에 집중
  c = 0.8×h₁ + 0.1×h₂ + 0.1×h₃

"student" 생성 시:
  α = [0.1, 0.8, 0.1]  → "학생"에 집중
  c = 0.1×h₁ + 0.8×h₂ + 0.1×h₃
```

### 3.5 Attention의 장점

```
1. 정보 병목 해결
   → 매번 원문 전체 참조 가능

2. 해석 가능성
   → Attention 가중치로 "어디를 봤는지" 시각화

3. 장거리 의존성
   → 먼 단어도 직접 연결
```

---

## 4. Self-Attention

### 4.1 Attention vs Self-Attention

```
Attention:
  Encoder ↔ Decoder 사이

Self-Attention:
  같은 시퀀스 내부에서
  "나는 학생이다" 각 단어가 서로 참조
```

### 4.2 왜 필요한가?

```
문장: "그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤했기 때문이다."

"그것" = ?
→ "동물"을 가리킴
→ Self-Attention으로 연결
```

### 4.3 Query, Key, Value

```
각 단어에서 3개 벡터 생성:
  Query (Q): "나는 무엇을 찾고 있는가?"
  Key (K):   "나는 어떤 정보를 가지고 있는가?"
  Value (V): "실제 전달할 정보"

Attention(Q, K, V) = softmax(Q·Kᵀ/√d) · V
```

### 4.4 계산 과정

```
입력: X = [x₁, x₂, x₃] (3개 단어)

Step 1: Q, K, V 생성
  Q = X · Wq
  K = X · Wk
  V = X · Wv

Step 2: 유사도 계산
  scores = Q · Kᵀ / √d

Step 3: Softmax
  weights = softmax(scores)

Step 4: 가중 평균
  output = weights · V
```

---

## 5. Transformer

### 5.1 핵심 아이디어

```
"Attention is All You Need" (2017)

RNN 없이, Attention만으로 시퀀스 처리
→ 병렬 처리 가능
→ 훨씬 빠름
```

### 5.2 전체 구조

```
        ┌─────────────────┐
        │     Outputs     │
        │   (shifted)     │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │    Decoder      │
        │  (N layers)     │
        └────────┬────────┘
                 ↑
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
┌───────────────┐ ┌───────────────┐
│   Encoder     │ │   Encoder     │
│  (N layers)   │ │   Output      │
└───────┬───────┘ └───────────────┘
        ↑
┌───────────────┐
│    Inputs     │
└───────────────┘
```

### 5.3 Encoder Block

```
┌─────────────────────────────────────┐
│                                     │
│  Input                              │
│    ↓                                │
│  ┌─────────────────────────┐        │
│  │  Multi-Head Attention   │        │
│  └───────────┬─────────────┘        │
│              ↓                      │
│  ┌─────────────────────────┐        │
│  │   Add & Norm            │←─────┐ │
│  └───────────┬─────────────┘      │ │
│              ↓                    │ │
│  ┌─────────────────────────┐      │ │
│  │   Feed Forward          │      │ │
│  └───────────┬─────────────┘      │ │
│              ↓                    │ │
│  ┌─────────────────────────┐      │ │
│  │   Add & Norm            │←─────┘ │
│  └───────────┬─────────────┘        │
│              ↓                      │
│           Output                    │
│                                     │
└─────────────────────────────────────┘
```

### 5.4 Multi-Head Attention

```
"여러 관점에서 동시에 바라보기"

Head 1: 문법적 관계 포착
Head 2: 의미적 관계 포착
Head 3: 위치적 관계 포착
...

MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ

headᵢ = Attention(Q·Wᵢᑫ, K·Wᵢᴷ, V·Wᵢⱽ)
```

### 5.5 Positional Encoding

```
문제: Self-Attention은 순서 정보 없음
     "나는 사과를 먹었다" = "먹었다 사과를 나는" (같은 결과)

해결: 위치 정보 추가

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

입력 = 단어 임베딩 + Positional Encoding
```

### 5.6 Decoder Block

```
Encoder Block + 추가:

1. Masked Self-Attention
   → 미래 단어 참조 방지
   → "I am" 생성 시 "a student" 못 봄

2. Encoder-Decoder Attention
   → Encoder 출력 참조
   → 번역 시 원문 참조
```

---

## 6. Transformer 수식 요약

### 6.1 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

dₖ = Key 벡터 차원
√dₖ로 나누는 이유: 내적 값이 커지면 softmax 기울기 소실
```

### 6.2 Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ

headᵢ = Attention(QWᵢᑫ, KWᵢᴷ, VWᵢⱽ)
```

### 6.3 Feed-Forward Network

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

= ReLU 활성화 함수를 가진 2층 신경망
```

### 6.4 Layer Normalization

```
LayerNorm(x) = γ × (x - μ) / σ + β

μ, σ = x의 평균, 표준편차
γ, β = 학습 가능한 모수
```

---

## 7. BERT와 GPT

### 7.1 Transformer 기반 사전학습 모델

```
Transformer (2017)
    ↓
BERT (2018)     GPT (2018)
Encoder만 사용   Decoder만 사용
양방향           단방향
이해에 강함      생성에 강함
```

### 7.2 BERT (Bidirectional Encoder Representations from Transformers)

```
학습 방법: Masked Language Model

"나는 [MASK] 먹었다"
→ [MASK] = "사과를" 예측

양방향:
  "나는"과 "먹었다" 모두 참조
  → 문맥 이해에 강함
```

### 7.3 GPT (Generative Pre-trained Transformer)

```
학습 방법: Next Token Prediction

"나는 사과를" → "먹었다" 예측

단방향:
  이전 단어만 참조
  → 텍스트 생성에 강함
```

### 7.4 비교

|항목|BERT|GPT|
|:--|:--|:--|
|구조|Encoder|Decoder|
|방향|양방향|단방향|
|학습|Masked LM|Next Token|
|강점|이해, 분류|생성|
|활용|QA, NER, 감성분석|대화, 글쓰기|

---

## 8. 시계열 예측에의 적용

### 8.1 Transformer for Time Series

```
기존: RNN/LSTM 순차 처리
     → 느림, 장기 의존성 한계

Transformer:
     → 병렬 처리 가능
     → Self-Attention으로 장기 의존성 포착
```

### 8.2 간단한 구조

```
시계열: [y₁, y₂, ..., yₜ]
     ↓
Positional Encoding 추가
     ↓
Transformer Encoder
     ↓
Linear Layer
     ↓
예측: ŷₜ₊₁
```

### 8.3 Python 코드 (PyTorch)

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.output_fc(x[:, -1, :])  # 마지막 시점만
        return out

# 모델 생성
model = TransformerPredictor(
    input_dim=1, 
    d_model=64, 
    nhead=4, 
    num_layers=2, 
    output_dim=1
)
```

---

## 9. 요약

### 9.1 Seq2Seq

```
Encoder-Decoder 구조
입력 시퀀스 → Context Vector → 출력 시퀀스
문제: 정보 병목
```

### 9.2 Attention

```
매번 원문 참조
"어디를 볼까?" 가중치로 결정
정보 병목 해결, 해석 가능
```

### 9.3 Transformer

```
Self-Attention 기반
RNN 없이 시퀀스 처리
병렬 처리 → 빠름
BERT, GPT의 기반
```

### 9.4 발전 흐름

```
RNN (1990) → LSTM (1997) → Seq2Seq (2014)
    → Attention (2015) → Transformer (2017)
        → BERT, GPT (2018) → GPT-4, Claude (2023~)
```

---

## 참고문헌

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. _NeurIPS_.
    
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. _ICLR_.
    
3. Vaswani, A., et al. (2017). Attention is all you need. _NeurIPS_.
    
4. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. _NAACL_.
    
5. Radford, A., et al. (2018). Improving language understanding by generative pre-training. _OpenAI_.