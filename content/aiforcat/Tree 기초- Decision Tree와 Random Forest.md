

## 1. 왜 Tree인가?

### 1.1 회귀 모형의 한계

```
선형 회귀: y = β₀ + β₁x₁ + β₂x₂ + ε

가정: y와 x의 관계가 선형
현실: 비선형 관계가 많음
```

**예시: 집값 예측**

```
면적 30평 이하: 평당 1000만원
면적 30평 초과: 평당 800만원 (대형 할인)

→ 선형 모형으로 표현 어려움
→ "조건"에 따라 다른 관계
```

### 1.2 Tree의 아이디어

```
"조건에 따라 분기"

면적 ≤ 30평?
    ├── Yes → 평당 1000만원
    └── No  → 평당 800만원
```

---

## 2. Decision Tree 구조

### 2.1 기본 용어

```
          [Root Node]
         면적 ≤ 30?
          /       \
        Yes        No
        /           \
   [Internal]    [Leaf]
   역세권?        2.4억
    /    \
  Yes    No
  /        \
[Leaf]   [Leaf]
3.5억     2.8억
```

|용어|설명|
|:--|:--|
|Root Node|최상위 노드, 첫 번째 분기|
|Internal Node|중간 노드, 추가 분기|
|Leaf Node|말단 노드, 최종 예측값|
|Depth|트리의 깊이 (분기 횟수)|
|Split|분기 기준|

### 2.2 수식 표현

**예측 함수:**

```
ŷ = f(x) = Σⱼ cⱼ · I(x ∈ Rⱼ)

Rⱼ = j번째 영역 (leaf node)
cⱼ = j번째 영역의 예측값
I() = 지시 함수 (해당 영역이면 1, 아니면 0)
```

**예시:**

```
R₁ = {면적 ≤ 30, 역세권}     → c₁ = 3.5억
R₂ = {면적 ≤ 30, 비역세권}   → c₂ = 2.8억
R₃ = {면적 > 30}            → c₃ = 2.4억

x = (면적=25, 역세권=Yes)
→ x ∈ R₁
→ ŷ = 3.5억
```

---

## 3. 모수: 무엇을 추정하는가?

### 3.1 두 가지 모수

**1) 분기 기준 (Split Criteria)**

```
어떤 변수로? (면적? 역세권?)
어떤 값에서? (30평? 25평?)
```

**2) 각 영역의 예측값 (Leaf Values)**

```
cⱼ = 영역 Rⱼ의 대표값

회귀: 평균
분류: 최빈값 (또는 확률)
```

### 3.2 모수 개수

```
Leaf 노드 수 = J
분기 수 = J - 1

모수:
- 분기 변수 선택: J - 1개
- 분기 임계값: J - 1개
- Leaf 예측값: J개

총: 3J - 2개 (대략)
```

---

## 4. 추정 방법: 재귀적 분할

### 4.1 핵심 질문

```
"어떤 변수의 어떤 값에서 나누면 좋은가?"
```

### 4.2 분할 기준 (회귀)

**RSS (Residual Sum of Squares) 최소화:**

```
RSS = Σᵢ∈R₁ (yᵢ - ĉ₁)² + Σᵢ∈R₂ (yᵢ - ĉ₂)²

ĉ₁ = R₁ 영역 y의 평균
ĉ₂ = R₂ 영역 y의 평균
```

**모든 가능한 분할에 대해:**

```
for 각 변수 j:
    for 각 분할점 s:
        R₁ = {x | xⱼ ≤ s}
        R₂ = {x | xⱼ > s}
        RSS 계산
        
최소 RSS를 주는 (j, s) 선택
```

### 4.3 분할 기준 (분류)

**Gini 불순도:**

```
Gini(R) = 1 - Σₖ p̂ₖ²

p̂ₖ = 영역 R에서 클래스 k의 비율
```

**예시:**

```
영역 R: [A, A, A, B] (4개 샘플)
p̂_A = 3/4 = 0.75
p̂_B = 1/4 = 0.25

Gini = 1 - (0.75² + 0.25²)
     = 1 - (0.5625 + 0.0625)
     = 0.375
```

**Gini 감소량 최대화:**

```
ΔGini = Gini(부모) - [n₁/n × Gini(R₁) + n₂/n × Gini(R₂)]
```

### 4.4 알고리즘: CART

```python
def build_tree(data, depth=0, max_depth=5):
    # 종료 조건
    if depth >= max_depth or len(data) < min_samples:
        return LeafNode(mean(data.y))  # 또는 mode for 분류
    
    # 최적 분할 찾기
    best_var, best_split, best_score = None, None, inf
    
    for var in variables:
        for split in possible_splits(data[var]):
            left = data[data[var] <= split]
            right = data[data[var] > split]
            score = compute_impurity(left, right)
            
            if score < best_score:
                best_var, best_split, best_score = var, split, score
    
    # 분할 수행
    left_data = data[data[best_var] <= best_split]
    right_data = data[data[best_var] > best_split]
    
    return InternalNode(
        var=best_var,
        split=best_split,
        left=build_tree(left_data, depth+1),
        right=build_tree(right_data, depth+1)
    )
```

---

## 5. 과적합 문제

### 5.1 문제 상황

```
깊은 트리:
- 학습 데이터 완벽히 맞춤
- 새 데이터에 일반화 안 됨

극단적 경우:
- 각 샘플마다 하나의 leaf
- 학습 오차 = 0
- 테스트 오차 = 매우 큼
```

### 5.2 해결: 가지치기 (Pruning)

**사전 가지치기 (Pre-pruning):**

```
- max_depth: 최대 깊이 제한
- min_samples_split: 분할 최소 샘플 수
- min_samples_leaf: leaf 최소 샘플 수
```

**사후 가지치기 (Post-pruning):**

```
1. 트리를 끝까지 성장
2. 아래에서부터 가지 제거
3. 검증 오차가 증가하면 중단

비용 복잡도 가지치기:
Cost = RSS + α × |T|

α = 복잡도 패널티
|T| = leaf 노드 수
```

---

## 6. 시뮬레이션: 참 모수 복원

### 6.1 참 모형 설정

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 참 모형: 2개 영역
# x ≤ 0.5: y = 2
# x > 0.5: y = 5

np.random.seed(42)
n = 100
x = np.random.uniform(0, 1, n)
y_true = np.where(x <= 0.5, 2, 5)
y = y_true + np.random.normal(0, 0.3, n)  # 노이즈 추가

X = x.reshape(-1, 1)
```

### 6.2 Tree 학습

```python
# Decision Tree 학습
tree = DecisionTreeRegressor(max_depth=1)
tree.fit(X, y)

# 결과 확인
print(f"분할점: {tree.tree_.threshold[0]:.3f}")  # 약 0.5
print(f"왼쪽 예측값: {tree.tree_.value[1][0][0]:.3f}")  # 약 2
print(f"오른쪽 예측값: {tree.tree_.value[2][0][0]:.3f}")  # 약 5
```

### 6.3 시각화

```python
# 트리 구조 시각화
plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=['x'], filled=True, rounded=True)
plt.title('학습된 Decision Tree')
plt.show()

# 예측 결과 시각화
x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = tree.predict(x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='데이터')
plt.plot(x_plot, y_pred, 'r-', linewidth=2, label='예측')
plt.axvline(x=0.5, color='g', linestyle='--', label='참 분할점')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Decision Tree 예측')
plt.show()
```

### 6.4 결과 확인

```
참 모수:
  분할점 = 0.5
  왼쪽 예측값 = 2
  오른쪽 예측값 = 5

추정 모수:
  분할점 ≈ 0.498
  왼쪽 예측값 ≈ 1.98
  오른쪽 예측값 ≈ 5.02

→ 참 모수 잘 복원됨!
```

---

## 7. Decision Tree의 한계

### 7.1 높은 분산 (High Variance)

```
데이터가 조금만 바뀌어도
→ 트리 구조가 크게 달라짐
→ 예측이 불안정

원인: 탐욕적(greedy) 알고리즘
     - 각 단계에서 최선만 선택
     - 전체 최적해 보장 안 됨
```

### 7.2 예시

```
데이터셋 A로 학습:
          면적 ≤ 30?
          /       \
        역세권?    2.4억
        /    \
      3.5억  2.8억

데이터셋 B로 학습 (샘플 조금 다름):
          역세권?
          /       \
      면적≤25?    면적≤35?
       /   \       /   \
     ...   ...   ...   ...

→ 구조가 완전히 다름!
```

### 7.3 해결: 앙상블

```
"여러 트리의 예측을 합치면 안정적"

방법 1: Bagging (Bootstrap Aggregating)
방법 2: Boosting
```

---

## 8. Random Forest

### 8.1 핵심 아이디어

```
여러 Decision Tree의 평균

1. 데이터 Bootstrap 샘플링
2. 변수 Random 선택
3. 각각 Tree 학습
4. 예측값 평균 (회귀) 또는 투표 (분류)
```

### 8.2 Bagging (Bootstrap Aggregating)

**Bootstrap 샘플:**

```
원본 데이터: [1, 2, 3, 4, 5] (n=5)

Bootstrap 샘플 1: [1, 1, 3, 4, 5] (복원 추출)
Bootstrap 샘플 2: [2, 2, 3, 5, 5]
Bootstrap 샘플 3: [1, 3, 3, 4, 4]
...
```

**Bagging 예측:**

```
ŷ_bagging = (1/B) × Σᵦ ŷᵦ

B = 트리 개수
ŷᵦ = b번째 트리의 예측
```

### 8.3 변수 무작위 선택

```
일반 Tree: 모든 변수 중 최적 선택
Random Forest: 일부 변수만 후보로

분기마다:
  - 전체 p개 변수 중 m개만 무작위 선택
  - m개 중에서 최적 분할 찾기

권장값:
  - 회귀: m = p/3
  - 분류: m = √p
```

**왜 변수를 제한하나?**

```
모든 변수 사용 시:
  - 강한 변수가 항상 선택됨
  - 모든 트리가 비슷해짐
  - 트리 간 상관관계 높음
  - 분산 감소 효과 적음

변수 제한 시:
  - 다양한 변수 사용
  - 트리 간 상관관계 낮음
  - 앙상블 효과 극대화
```

### 8.4 알고리즘

```python
def random_forest(data, n_trees=100, max_features='sqrt'):
    trees = []
    
    for b in range(n_trees):
        # Bootstrap 샘플
        boot_idx = np.random.choice(len(data), len(data), replace=True)
        boot_data = data[boot_idx]
        
        # Tree 학습 (변수 제한)
        tree = DecisionTree(max_features=max_features)
        tree.fit(boot_data)
        trees.append(tree)
    
    return trees

def predict(trees, x):
    predictions = [tree.predict(x) for tree in trees]
    return np.mean(predictions)  # 회귀
    # return mode(predictions)  # 분류
```

### 8.5 Out-of-Bag (OOB) 오차

```
Bootstrap 샘플에 포함되지 않은 데이터
→ 약 37%의 데이터가 각 트리에서 제외
→ 이 데이터로 검증 가능

OOB 오차:
  각 샘플 i에 대해:
    - i를 포함하지 않은 트리들로 예측
    - 실제값과 비교

→ Cross-validation 없이 검증 가능!
```

---

## 9. Random Forest 실습

### 9.1 Python 코드

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 생성 (비선형 관계)
np.random.seed(42)
n = 500
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

# 참 모형: y = sin(x1) + 0.5*x2 + noise
y = np.sin(x1) + 0.5 * x2 + np.random.normal(0, 0.5, n)

X = np.column_stack([x1, x2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest 학습
rf = RandomForestRegressor(
    n_estimators=100,      # 트리 개수
    max_depth=10,          # 최대 깊이
    max_features='sqrt',   # 변수 선택
    oob_score=True,        # OOB 점수 계산
    random_state=42
)
rf.fit(X_train, y_train)

# 결과
print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, rf.predict(X_test))):.4f}")
```

### 9.2 변수 중요도

```python
# 변수 중요도 (불순도 감소 기반)
importance = rf.feature_importances_
print(f"x1 중요도: {importance[0]:.4f}")
print(f"x2 중요도: {importance[1]:.4f}")

# 시각화
plt.figure(figsize=(8, 4))
plt.bar(['x1', 'x2'], importance)
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### 9.3 트리 개수에 따른 성능

```python
n_trees_list = [1, 5, 10, 50, 100, 200]
oob_scores = []

for n_trees in n_trees_list:
    rf = RandomForestRegressor(n_estimators=n_trees, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)

plt.figure(figsize=(10, 6))
plt.plot(n_trees_list, oob_scores, 'bo-')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('Performance vs Number of Trees')
plt.show()

# 결과: 트리 개수 증가 → 성능 향상 → 수렴
```

---

## 10. 요약

### 10.1 Decision Tree

```
구조: 조건에 따른 분기
모수: 분할 기준 + Leaf 예측값
추정: RSS/Gini 최소화 (탐욕적)
장점: 해석 가능, 비선형 포착
단점: 과적합, 높은 분산
```

### 10.2 Random Forest

```
구조: 여러 Decision Tree의 앙상블
핵심:
  1. Bootstrap 샘플링
  2. 변수 무작위 선택
  3. 예측값 평균/투표

장점: 낮은 분산, 과적합 완화
단점: 해석 어려움, 느림 (많은 트리)
```

### 10.3 다음 강의 예고

```
Random Forest: 독립적인 트리들의 평균
Boosting: 순차적으로 트리 개선

→ XGBoost, LightGBM, MTGBM
```

---

## 참고문헌

1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). _Classification and regression trees_. CRC press.
    
2. Breiman, L. (2001). Random forests. _Machine learning_, 45(1), 5-32.
    
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The elements of statistical learning_. Springer.