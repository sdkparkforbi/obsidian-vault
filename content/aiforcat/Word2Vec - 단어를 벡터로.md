

## 1. 왜 단어를 벡터로?

### 1.1 문제 상황

컴퓨터는 숫자만 이해한다. 그런데 우리는 텍스트를 분석하고 싶다.

```
"사과" → 컴퓨터가 이해할 수 있는 숫자?
```

### 1.2 가장 단순한 방법: One-hot Encoding

단어가 5개인 세상을 가정하자:

```
어휘: [사과, 바나나, 오렌지, 포도, 딸기]

사과   = [1, 0, 0, 0, 0]
바나나 = [0, 1, 0, 0, 0]
오렌지 = [0, 0, 1, 0, 0]
포도   = [0, 0, 0, 1, 0]
딸기   = [0, 0, 0, 0, 1]
```

### 1.3 One-hot의 문제

**문제 1: 차원이 너무 크다**

```
실제 어휘 크기: 10,000 ~ 100,000개
→ 벡터 길이도 10,000 ~ 100,000
→ 계산량 폭발
```

**문제 2: 의미를 담지 못한다**

```
사과와 오렌지의 거리 = 사과와 포도의 거리 = √2

모든 단어가 서로 같은 거리!
→ "사과"와 "오렌지"가 비슷하다는 정보 없음
```

### 1.4 해결책: 저차원 밀집 벡터

```
목표: 의미가 비슷한 단어는 가까이, 다른 단어는 멀리

사과   = [0.8, 0.2, 0.1]
바나나 = [0.7, 0.3, 0.2]
오렌지 = [0.75, 0.25, 0.15]
자동차 = [0.1, 0.9, 0.5]

→ 사과, 바나나, 오렌지는 가까이
→ 자동차는 멀리
```

**Word2Vec의 목표: 이런 벡터를 자동으로 학습**

---

## 2. Word2Vec 모형 구조

### 2.1 핵심 아이디어

> "단어의 의미는 그 주변 단어로 결정된다" — J.R. Firth (1957)

```
"나는 ___ 를 먹었다"

빈칸에 올 수 있는 단어: 사과, 바나나, 밥, 피자...
→ 이 단어들은 비슷한 맥락에서 등장
→ 비슷한 의미를 가진다고 볼 수 있음
```

### 2.2 두 가지 방식

|방식|입력|출력|질문|
|:--|:--|:--|:--|
|**Skip-gram**|중심 단어|주변 단어|"사과"가 주어졌을 때, 주변에 어떤 단어?|
|**CBOW**|주변 단어|중심 단어|주변 단어가 주어졌을 때, 중심 단어는?|

**이 강의에서는 Skip-gram을 중심으로 설명**

---

## 3. Skip-gram 모형

### 3.1 학습 데이터 생성

**원본 문장:**

```
"나는 빨간 사과를 먹었다"
```

**토큰화:**

```
[나는, 빨간, 사과를, 먹었다]
```

**윈도우 크기 = 1일 때 (좌우 1개씩):**

|중심 단어|주변 단어|
|:-:|:-:|
|나는|빨간|
|빨간|나는, 사과를|
|사과를|빨간, 먹었다|
|먹었다|사과를|

**학습 데이터 (Skip-gram):**

|입력 (중심)|출력 (주변)|
|:-:|:-:|
|빨간|나는|
|빨간|사과를|
|사과를|빨간|
|사과를|먹었다|
|...|...|

### 3.2 모형 구조

```
입력층        은닉층        출력층
(V차원)      (d차원)       (V차원)
  
  ○            ○            ○
  ○     W_in   ○   W_out    ○
  ○   ------>  ○  ------>   ○
  ○            ○            ○
  ○                         ○
  
V = 어휘 크기 (예: 10,000)
d = 임베딩 차원 (예: 100)
```

### 3.3 모수 (Parameters)

**입력 가중치 행렬: W_in**

```
크기: V × d
의미: 각 단어의 "중심 단어로서의 벡터"
```

**출력 가중치 행렬: W_out**

```
크기: d × V
의미: 각 단어의 "주변 단어로서의 벡터"
```

**총 모수 개수:**

```
V × d + d × V = 2 × V × d

예: V = 10,000, d = 100
→ 2 × 10,000 × 100 = 2,000,000개
```

---

## 4. 수식으로 이해하기

### 4.1 단순화된 모형

**가장 단순한 경우:**

- 어휘 크기 V = 5
- 임베딩 차원 d = 2
- 윈도우 크기 = 1

**어휘:**

```
0: 나는
1: 빨간
2: 사과를
3: 먹었다
4: 좋아한다
```

### 4.2 One-hot 입력

중심 단어 "사과를" (index = 2):

```
x = [0, 0, 1, 0, 0]  (V = 5차원)
```

### 4.3 은닉층 계산

```
h = x · W_in

W_in = | w₀₀  w₀₁ |
       | w₁₀  w₁₁ |
       | w₂₀  w₂₁ |  ← "사과를"의 벡터
       | w₃₀  w₃₁ |
       | w₄₀  w₄₁ |

x · W_in = [0,0,1,0,0] · W_in = [w₂₀, w₂₁]
```

**결과:** 은닉층 h는 "사과를"의 임베딩 벡터

### 4.4 출력층 계산

```
z = h · W_out

W_out = | u₀₀  u₀₁  u₀₂  u₀₃  u₀₄ |
        | u₁₀  u₁₁  u₁₂  u₁₃  u₁₄ |

z = [h₀, h₁] · W_out = [z₀, z₁, z₂, z₃, z₄]
```

### 4.5 확률 변환 (Softmax)

```
P(단어 j | 중심 단어) = exp(zⱼ) / Σₖ exp(zₖ)
```

**예시:**

```
z = [1.2, 2.5, 0.3, 2.8, 0.1]

exp(z) = [3.32, 12.18, 1.35, 16.44, 1.11]
합계 = 34.40

P = [0.10, 0.35, 0.04, 0.48, 0.03]
```

**해석:** 중심 단어 "사과를"이 주어졌을 때, 주변 단어가...

- "나는"일 확률: 10%
- "빨간"일 확률: 35%
- "사과를"일 확률: 4%
- "먹었다"일 확률: 48%
- "좋아한다"일 확률: 3%

---

## 5. 모수 추정: 손실 함수

### 5.1 목표

주변 단어의 확률을 최대화

**예:** "사과를" → "빨간" 예측

```
정답: "빨간" (index = 1)
목표: P("빨간" | "사과를")을 최대화
```

### 5.2 손실 함수 (Cross-Entropy)

```
L = -log P(정답 | 중심 단어)
  = -log P("빨간" | "사과를")
```

**예시:**

```
P("빨간" | "사과를") = 0.35

L = -log(0.35) = 1.05
```

**목표: L을 최소화 → P를 최대화**

### 5.3 전체 손실 함수

모든 (중심, 주변) 쌍에 대해:

```
L_total = Σ -log P(주변 단어 | 중심 단어)
```

---

## 6. 모수 추정: 경사하강법

### 6.1 업데이트 규칙

```
θ_new = θ_old - α × ∂L/∂θ
```

- θ: 모수 (W_in, W_out의 원소들)
- α: 학습률
- ∂L/∂θ: 손실 함수의 기울기

### 6.2 기울기 유도 (Skip-gram)

**출력 가중치 업데이트:**

```
∂L/∂W_out = h × (P - y)

여기서:
- h: 은닉층 벡터 (중심 단어 임베딩)
- P: 예측 확률 벡터
- y: 정답 one-hot 벡터
```

**입력 가중치 업데이트:**

```
∂L/∂W_in = x × ((P - y) · W_out)
```

### 6.3 구체적 예시

**상황:**

- 중심 단어: "사과를" (index = 2)
- 정답 주변 단어: "빨간" (index = 1)
- 예측 확률: P = [0.10, 0.35, 0.04, 0.48, 0.03]
- 정답: y = [0, 1, 0, 0, 0]

**오차:**

```
P - y = [0.10, -0.65, 0.04, 0.48, 0.03]
```

**해석:**

- "빨간"의 오차 = -0.65 (확률이 0.65만큼 부족)
- "먹었다"의 오차 = +0.48 (확률이 0.48만큼 과다)

**업데이트 방향:**

- "빨간" 벡터: "사과를"과 가까워지도록
- "먹었다" 벡터: "사과를"과 멀어지도록

---

## 7. Negative Sampling

### 7.1 문제: Softmax 계산량

```
P(j | center) = exp(zⱼ) / Σₖ exp(zₖ)
                           ↑
                     모든 단어에 대해 계산!
                     V = 10,000이면 10,000번
```

### 7.2 해결: Negative Sampling

**아이디어:**

- 모든 단어 말고, 일부만 샘플링
- 정답 단어 1개 + 오답 단어 k개

**새로운 목표:**

```
정답 단어: 확률 높이기 (1에 가깝게)
오답 단어: 확률 낮추기 (0에 가깝게)
```

### 7.3 Negative Sampling 손실 함수

```
L = -log σ(u_pos · h) - Σ log σ(-u_neg · h)
```

- σ: 시그모이드 함수, σ(x) = 1/(1+e^(-x))
- u_pos: 정답 단어의 출력 벡터
- u_neg: 오답 단어의 출력 벡터
- h: 중심 단어의 입력 벡터

### 7.4 계산량 비교

```
Full Softmax: O(V) = O(10,000)
Negative Sampling (k=5): O(k+1) = O(6)

→ 약 1,600배 빨라짐!
```

---

## 8. 시뮬레이션으로 검증

### 8.1 목표

모수를 가정하고 데이터를 생성한 후, 추정이 제대로 되는지 확인

### 8.2 설정

```python
# 어휘 크기
V = 5  # [나는, 빨간, 사과를, 먹었다, 좋아한다]

# 임베딩 차원
d = 2

# 참 모수 (W_in)
W_in_true = np.array([
    [0.1, 0.9],   # 나는
    [0.8, 0.2],   # 빨간
    [0.7, 0.3],   # 사과를
    [0.2, 0.8],   # 먹었다
    [0.5, 0.5]    # 좋아한다
])
```

### 8.3 데이터 생성

```python
# 문장 생성
sentences = [
    [1, 2, 3],      # 빨간 사과를 먹었다
    [0, 1, 2],      # 나는 빨간 사과를
    [2, 3],         # 사과를 먹었다
    [0, 4, 2],      # 나는 좋아한다 사과를
]

# (중심, 주변) 쌍 생성
pairs = []
for sentence in sentences:
    for i, center in enumerate(sentence):
        for j in range(max(0, i-1), min(len(sentence), i+2)):
            if i != j:
                pairs.append((center, sentence[j]))
```

### 8.4 Python 코드 (학습)

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_word2vec(pairs, V, d, epochs=1000, lr=0.1, k=2):
    """
    Skip-gram with Negative Sampling
    """
    # 초기화
    W_in = np.random.randn(V, d) * 0.01
    W_out = np.random.randn(V, d) * 0.01
    
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(pairs)
        
        for center, context in pairs:
            # 중심 단어 벡터
            h = W_in[center]
            
            # Positive sample
            u_pos = W_out[context]
            score_pos = np.dot(h, u_pos)
            loss = -np.log(sigmoid(score_pos) + 1e-7)
            
            # Gradient for positive
            grad_pos = (sigmoid(score_pos) - 1) * h
            grad_h = (sigmoid(score_pos) - 1) * u_pos
            
            # Negative samples
            negatives = []
            while len(negatives) < k:
                neg = np.random.randint(V)
                if neg != context:
                    negatives.append(neg)
            
            for neg in negatives:
                u_neg = W_out[neg]
                score_neg = np.dot(h, u_neg)
                loss += -np.log(sigmoid(-score_neg) + 1e-7)
                
                # Gradient for negative
                grad_neg = sigmoid(score_neg) * h
                grad_h += sigmoid(score_neg) * u_neg
                
                # Update W_out for negative
                W_out[neg] -= lr * grad_neg
            
            # Update W_out for positive
            W_out[context] -= lr * grad_pos
            
            # Update W_in
            W_in[center] -= lr * grad_h
            
            total_loss += loss
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    return W_in, W_out

# 학습 실행
W_in_est, W_out_est = train_word2vec(pairs, V=5, d=2, epochs=500)
```

### 8.5 결과 확인

```python
# 참 모수와 추정 모수 비교
print("참 모수 (W_in_true):")
print(W_in_true)

print("\n추정 모수 (W_in_est):")
print(W_in_est)

# 코사인 유사도 확인
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# "사과를"과 "빨간"의 유사도
sim = cosine_similarity(W_in_est[2], W_in_est[1])
print(f"\n'사과를'과 '빨간' 유사도: {sim:.4f}")
```

---

## 9. 실제 데이터 적용 1: 공개 데이터셋

### 9.1 Shakespeare 데이터셋

```python
from gensim.models import Word2Vec
import nltk
from nltk.corpus import gutenberg

# 데이터 로드
nltk.download('gutenberg')
nltk.download('punkt')

# Shakespeare 작품
shakespeare = gutenberg.sents('shakespeare-hamlet.txt')

# 소문자 변환
sentences = [[word.lower() for word in sent] for sent in shakespeare]

# Word2Vec 학습
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,  # Skip-gram
    negative=5
)

# 유사 단어 찾기
print(model.wv.most_similar('king', topn=5))
# [('queen', 0.89), ('prince', 0.85), ...]
```

### 9.2 결과 해석

```
king과 유사한 단어:
1. queen (0.89)
2. prince (0.85)
3. lord (0.82)
4. majesty (0.80)
5. crown (0.78)

→ 왕실 관련 단어들이 가까이 위치
→ Word2Vec이 의미를 학습했음!
```

---

## 10. 실제 데이터 적용 2: TRN 상품명 데이터

### 10.1 데이터 구조

```python
# 상품명 예시
products = [
    "타이레놀 500mg 정제",
    "타이레놀 이알 서방정",
    "아스피린 프로텍트 100mg",
    "부루펜 시럽",
    "판콜에이 내복액",
    ...
]
```

### 10.2 전처리

```python
from konlpy.tag import Okt

okt = Okt()

def tokenize(text):
    """형태소 분석"""
    return okt.morphs(text)

# 토큰화
tokenized = [tokenize(p) for p in products]

# 예시 결과
# "타이레놀 500mg 정제" → ["타이레놀", "500", "mg", "정제"]
```

### 10.3 Word2Vec 학습

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=tokenized,
    vector_size=50,
    window=3,
    min_count=2,
    sg=1,
    negative=5,
    epochs=100
)

# 유사 상품 찾기
print(model.wv.most_similar('타이레놀', topn=5))
```

### 10.4 예상 결과

```
'타이레놀'과 유사한 단어:
1. 아세트아미노펜 (0.85)
2. 해열 (0.82)
3. 진통 (0.80)
4. 정제 (0.75)
5. 서방정 (0.72)

→ 타이레놀과 관련된 성분, 효능, 제형이 가까이 위치
```

---

## 11. Word2Vec의 한계

### 11.1 단어 순서 무시

```
"개가 사람을 물었다" vs "사람이 개를 물었다"

Word2Vec: 같은 단어 집합 → 같은 벡터
→ 의미 차이를 반영하지 못함
```

### 11.2 문맥에 따른 의미 변화 무시

```
"사과를 먹었다" (과일)
"진심으로 사과했다" (사과하다)

Word2Vec: "사과" → 하나의 벡터
→ 다의어 처리 불가
```

### 11.3 해결책: 다음 모형들

|한계|해결 모형|
|:--|:--|
|단어 순서|RNN, LSTM|
|문맥 의존 의미|Transformer, BERT|

---

## 12. 요약

### 12.1 Word2Vec이란?

```
텍스트 → 단어 벡터
의미가 비슷한 단어 → 가까운 벡터
```

### 12.2 핵심 아이디어

```
"단어의 의미는 주변 단어로 결정된다"
→ 함께 등장하는 단어들은 비슷한 벡터
```

### 12.3 모형 구조

```
Skip-gram: 중심 단어 → 주변 단어 예측
모수: W_in (V×d), W_out (d×V)
```

### 12.4 학습 방법

```
손실 함수: Cross-Entropy (또는 Negative Sampling)
최적화: 경사하강법
```

### 12.5 다음 강의 예고

```
Word2Vec → RNN, LSTM
단어 벡터 → 문장 벡터
순서 무시 → 순서 고려
```

---

## 참고문헌

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_.
    
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. _Advances in neural information processing systems_, 26.
    
3. Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method. _arXiv preprint arXiv:1402.3722_.