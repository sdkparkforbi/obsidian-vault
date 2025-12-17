

## 1. Bagging vs Boosting

### 1.1 복습: Bagging (Random Forest)

```
여러 트리를 독립적으로 학습
→ 예측값 평균

특징:
- 병렬 처리 가능
- 분산 감소
- 편향은 그대로
```

### 1.2 Boosting 아이디어

```
여러 트리를 순차적으로 학습
→ 이전 트리의 "실수"를 다음 트리가 보완

특징:
- 순차 처리 (병렬 어려움)
- 편향 감소
- 분산도 감소 가능
```

### 1.3 비교

|항목|Bagging|Boosting|
|:--|:--|:--|
|학습 방식|독립적 (병렬)|순차적|
|목표|분산 감소|편향 + 분산 감소|
|트리 구조|깊은 트리|얕은 트리 (약한 학습기)|
|샘플 가중치|동일|오차에 따라 조정|
|대표 알고리즘|Random Forest|XGBoost, LightGBM|

---

## 2. Gradient Boosting 원리

### 2.1 핵심 아이디어

```
"잔차(residual)를 예측하는 트리를 추가"

Step 1: 초기 예측 (평균)
Step 2: 잔차 계산
Step 3: 잔차를 예측하는 트리 학습
Step 4: 예측 업데이트
Step 5: 반복
```

### 2.2 수식

**초기화:**

```
F₀(x) = argmin_c Σᵢ L(yᵢ, c)

회귀 (MSE): F₀(x) = ȳ (평균)
```

**반복 (m = 1, 2, ..., M):**

```
Step 1: 잔차 계산 (음의 기울기)
  rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}

  MSE의 경우: rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)

Step 2: 잔차를 예측하는 트리 hₘ 학습
  hₘ = argmin_h Σᵢ (rᵢₘ - h(xᵢ))²

Step 3: 모델 업데이트
  Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)

  η = 학습률 (0 < η ≤ 1)
```

### 2.3 간단한 예시

```
데이터: x = [1, 2, 3], y = [2, 4, 6]

Step 0: 초기화
  F₀ = ȳ = 4

Step 1: 잔차 계산
  r = y - F₀ = [2-4, 4-4, 6-4] = [-2, 0, 2]

Step 2: 잔차 예측 트리 h₁ 학습
  h₁: x ≤ 2 → -1, x > 2 → 2

Step 3: 업데이트 (η = 0.5)
  F₁ = F₀ + 0.5 × h₁
  F₁(1) = 4 + 0.5×(-1) = 3.5
  F₁(2) = 4 + 0.5×(-1) = 3.5
  F₁(3) = 4 + 0.5×(2) = 5

Step 4: 새 잔차
  r = y - F₁ = [2-3.5, 4-3.5, 6-5] = [-1.5, 0.5, 1]

→ 계속 반복...
```

### 2.4 학습률의 역할

```
η = 1.0: 빠른 학습, 과적합 위험
η = 0.1: 느린 학습, 일반화 좋음

보통: η = 0.01 ~ 0.3
트리 개수와 트레이드오프:
  η 작으면 → 트리 많이 필요
  η 크면 → 트리 적게 필요
```

---

## 3. XGBoost (Extreme Gradient Boosting)

### 3.1 Gradient Boosting의 개선

```
XGBoost = Gradient Boosting + 정규화 + 최적화

핵심 개선:
1. 목적 함수에 정규화 항 추가
2. 2차 근사로 빠른 최적화
3. 효율적인 분할 탐색
4. 결측치 자동 처리
5. 병렬 처리 지원
```

### 3.2 목적 함수

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₘ Ω(fₘ)
      ───────────   ─────────
       손실 함수     정규화 항

정규화:
Ω(f) = γT + (1/2)λ Σⱼ wⱼ²

T = leaf 노드 수
wⱼ = j번째 leaf의 가중치
γ = 트리 복잡도 패널티
λ = L2 정규화 계수
```

### 3.3 2차 근사 (Taylor Expansion)

```
손실 함수의 2차 근사:

L(yᵢ, ŷᵢ + fₘ(xᵢ)) ≈ L(yᵢ, ŷᵢ) + gᵢfₘ(xᵢ) + (1/2)hᵢfₘ(xᵢ)²

gᵢ = ∂L/∂ŷᵢ     (1차 미분, gradient)
hᵢ = ∂²L/∂ŷᵢ²   (2차 미분, hessian)
```

**MSE의 경우:**

```
L = (yᵢ - ŷᵢ)²

gᵢ = -2(yᵢ - ŷᵢ) = -2 × 잔차
hᵢ = 2
```

### 3.4 최적 분할 찾기

**Gain (분할 이득):**

```
Gain = (1/2)[G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ

G = Σgᵢ (영역 내 gradient 합)
H = Σhᵢ (영역 내 hessian 합)
```

**분할 선택:**

```
모든 변수, 모든 분할점에 대해:
  Gain 계산
  최대 Gain의 분할 선택
  
Gain < 0 이면 분할 중단
```

### 3.5 Python 코드

https://colab.research.google.com/drive/1IRM-zZE03uJtdEaQmt-v4e30U0SRui1v?usp=sharing

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# 1. 예시 데이터 생성 (카페 매출 예측)
# ============================================
np.random.seed(42)
n_samples = 1000

X = pd.DataFrame({
    '기온': np.random.uniform(0, 35, n_samples),
    '강수량': np.random.uniform(0, 50, n_samples),
    '주말': np.random.randint(0, 2, n_samples),
    '프로모션': np.random.randint(0, 2, n_samples),
    '경쟁업체_거리': np.random.uniform(0.1, 5, n_samples)
})

# 실제 관계 + 노이즈
y = (100 
     + 2.5 * X['기온'] 
     - 1.8 * X['강수량'] 
     + 35 * X['주말'] 
     + 25 * X['프로모션']
     + 10 * X['경쟁업체_거리']
     + np.random.normal(0, 15, n_samples))

print(f"데이터 shape: X={X.shape}, y={y.shape}")
print(f"매출 범위: {y.min():.1f} ~ {y.max():.1f}")

# ============================================
# 2. 데이터 분할
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DMatrix 생성 (XGBoost 최적화 데이터 구조)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ============================================
# 3. 하이퍼파라미터 설정
# ============================================
params = {
    'objective': 'reg:squarederror',  # 손실 함수
    'max_depth': 6,                    # 트리 깊이
    'eta': 0.1,                        # 학습률
    'lambda': 1,                       # L2 정규화
    'gamma': 0,                        # 분할 최소 gain
    'subsample': 0.8,                  # 샘플 비율
    'colsample_bytree': 0.8,           # 변수 비율
    'eval_metric': 'rmse'
}

# ============================================
# 4. 모델 학습
# ============================================
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10,
    evals_result=evals_result
)

# ============================================
# 5. 예측 및 평가
# ============================================
y_pred = model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n최종 Test RMSE: {rmse:.4f}")

# ============================================
# 6. 시각화
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 학습 곡선
axes[0].plot(evals_result['train']['rmse'], label='Train')
axes[0].plot(evals_result['test']['rmse'], label='Test')
axes[0].set_xlabel('Boosting Round')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Learning Curve')
axes[0].legend()

# 변수 중요도
xgb.plot_importance(model, ax=axes[1], importance_type='gain')
axes[1].set_title('Feature Importance (Gain)')

# 실제 vs 예측
axes[2].scatter(y_test, y_pred, alpha=0.5)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title(f'Actual vs Predicted (RMSE={rmse:.2f})')

plt.tight_layout()
plt.show()
```

---
### 3.6 주요 하이퍼파라미터

| 파라미터                | 설명         | 권장값      |
| :------------------ | :--------- | :------- |
| max_depth           | 트리 최대 깊이   | 3~10     |
| eta (learning_rate) | 학습률        | 0.01~0.3 |
| n_estimators        | 트리 개수      | 100~1000 |
| lambda (reg_lambda) | L2 정규화     | 0~10     |
| gamma               | 분할 최소 gain | 0~5      |
| subsample           | 샘플 비율      | 0.5~1.0  |
| colsample_bytree    | 변수 비율      | 0.5~1.0  |

---

## 4. LightGBM

### 4.1 XGBoost의 한계

```
대용량 데이터에서:
- 모든 분할점 탐색 → 느림
- 메모리 사용량 많음
```

### 4.2 LightGBM의 핵심 개선

**1) Histogram-based 분할**

```
XGBoost: 모든 값에서 분할점 탐색
LightGBM: 값을 bin으로 묶어서 탐색

예:
연속 변수 x: [0.1, 0.3, 0.5, 0.7, 0.9, ...]
→ bin: [0, 1, 2, 3, 4, ...]  (256개 bin)

탐색 횟수: O(n) → O(256)
```

**2) Leaf-wise 성장**

```
Level-wise (XGBoost):
  깊이별로 모든 노드 분할
  → 균형 잡힌 트리
  → 불필요한 분할 포함

Leaf-wise (LightGBM):
  가장 큰 gain의 leaf만 분할
  → 비대칭 트리 가능
  → 효율적, 더 낮은 손실
```

**시각화:**

```
Level-wise:           Leaf-wise:
      ○                    ○
    /   \                /   \
   ○     ○              ○     ○
  / \   / \            /       \
 ○   ○ ○   ○          ○         ○
                     /
                    ○
```

**3) GOSS (Gradient-based One-Side Sampling)**

```
gradient가 큰 샘플: 정보량 많음 → 모두 사용
gradient가 작은 샘플: 정보량 적음 → 일부만 샘플링

→ 계산량 감소, 정확도 유지
```

**4) EFB (Exclusive Feature Bundling)**

```
상호 배타적인 변수들을 묶음
예: one-hot encoding된 변수들

[0,1,0,0] + [0,0,1,0] → 하나의 변수로

→ 변수 수 감소, 속도 향상
```

### 4.3 Python 코드

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dataset 생성
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 하이퍼파라미터
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',      # 또는 'dart', 'goss'
    'num_leaves': 31,              # leaf 최대 개수
    'max_depth': -1,               # -1 = 제한 없음
    'learning_rate': 0.1,
    'feature_fraction': 0.8,       # 변수 비율
    'bagging_fraction': 0.8,       # 샘플 비율
    'bagging_freq': 5,             # bagging 빈도
    'lambda_l1': 0,                # L1 정규화
    'lambda_l2': 0,                # L2 정규화
    'verbose': -1
}

# 학습
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
)

# 예측
y_pred = model.predict(X_test)
```

### 4.4 XGBoost vs LightGBM

|항목|XGBoost|LightGBM|
|:--|:--|:--|
|트리 성장|Level-wise|Leaf-wise|
|분할 탐색|정확 (모든 값)|근사 (Histogram)|
|속도|느림|빠름 (2~10배)|
|메모리|많음|적음|
|정확도|높음|비슷하거나 높음|
|과적합|덜 민감|민감 (조절 필요)|

---

## 5. MTGBM (Multi-Target Gradient Boosting Machine)

### 5.1 문제: 다중 출력 예측

```
일반 Boosting:
  X → y (단일 출력)

다중 출력:
  X → [y₁, y₂, ..., yₖ] (여러 출력)

예: 환자 데이터 → [Cmax, Tmax, AUC, Half-life]
```

### 5.2 기존 방법의 한계

```
방법 1: 출력별로 별도 모델
  - 출력 간 상관관계 무시
  - 비효율적

방법 2: 출력을 하나씩 순차 예측
  - 오차 누적
  - 순서 의존성
```

### 5.3 MTGBM 아이디어

```
여러 출력을 동시에 고려하는 Gradient Boosting

핵심:
1. 다중 출력 손실 함수
2. 출력 간 상관관계 학습
3. 공유 트리 구조
```

### 5.4 목적 함수

```
Obj = Σᵢ Σₖ L(yᵢₖ, ŷᵢₖ) + Ω(f)

i = 샘플 인덱스
k = 출력 인덱스

또는 가중 손실:
Obj = Σᵢ Σₖ wₖ × L(yᵢₖ, ŷᵢₖ)

wₖ = 출력 k의 가중치
```

### 5.5 구현 접근법

**방법 1: Multi-output Tree**

```
각 leaf에서 여러 값 출력:
  leaf j → [c_j1, c_j2, ..., c_jk]
```

**방법 2: Chained Boosting**

```
y₁ 예측 → y₂ 예측 (y₁ 포함) → y₃ 예측 (y₁, y₂ 포함)
```

**방법 3: 출력별 트리 + 공유 학습**

```
각 출력에 대해 트리 학습
but 잔차 계산 시 다른 출력 정보 활용
```

### 5.6 Python 코드 (scikit-learn 기반)

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# 다중 출력 데이터
# y: (n_samples, n_outputs)

# 방법 1: MultiOutputRegressor (독립 학습)
base_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
multi_model = MultiOutputRegressor(base_model)
multi_model.fit(X_train, y_train)

y_pred = multi_model.predict(X_test)
```

### 5.7 LightGBM으로 Multi-Target

```python
import lightgbm as lgb
import numpy as np

# 출력별 모델 학습 (공유 하이퍼파라미터)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.1,
}

models = []
for k in range(y_train.shape[1]):  # 각 출력에 대해
    train_data = lgb.Dataset(X_train, label=y_train[:, k])
    model = lgb.train(params, train_data, num_boost_round=100)
    models.append(model)

# 예측
y_pred = np.column_stack([m.predict(X_test) for m in models])
```

### 5.8 출력 상관관계 활용

```python
# Chained 방식: 이전 출력을 입력에 추가
from sklearn.multioutput import RegressorChain

chain_model = RegressorChain(
    GradientBoostingRegressor(n_estimators=100),
    order=[0, 1, 2, 3]  # 출력 예측 순서
)
chain_model.fit(X_train, y_train)

# 예측 시 y₁ → y₂ → y₃ → y₄ 순차적으로
y_pred = chain_model.predict(X_test)
```

---

## 6. 시뮬레이션: 참 모수 복원

### 6.1 설정

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# 데이터 생성 (이 부분이 먼저 실행되어야 함)
np.random.seed(42)
x = np.random.uniform(-3, 3, 500).reshape(-1, 1)
y_true = np.sin(x) + 0.5 * np.cos(2*x)  # 실제 함수
y = y_true.reshape(-1) + np.random.normal(0, 0.2, 500)  # 노이즈 추가

X = x  # 학습용
```

### 6.2 학습 및 비교

```python

# 다양한 트리 개수로 학습
n_estimators_list = [1, 5, 10, 50, 100]
x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true_plot = np.sin(x_plot) + 0.5 * np.cos(2*x_plot)  # ← x_plot 기준으로 계산

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, n_est in enumerate(n_estimators_list):
    model = GradientBoostingRegressor(
        n_estimators=n_est,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X, y)
    y_pred = model.predict(x_plot)
    
    axes[i].scatter(x, y, alpha=0.3, s=10)
    axes[i].plot(x_plot, y_true_plot, 'g-', label='True', linewidth=2)  # ← 수정
    axes[i].plot(x_plot, y_pred, 'r-', label='Pred', linewidth=2)
    axes[i].set_title(f'n_estimators = {n_est}')
    axes[i].legend()

plt.tight_layout()
plt.show()```

### 6.3 학습 곡선

```python
# 학습 과정 시각화
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
model.fit(X, y)

# 단계별 예측
train_scores = []
for i, y_pred in enumerate(model.staged_predict(X)):
    mse = np.mean((y - y_pred)**2)
    train_scores.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(train_scores)
plt.xlabel('Boosting Iteration')
plt.ylabel('MSE')
plt.title('Learning Curve')
plt.show()
```

---

## 7. 실제 적용: 약동학 모수 예측

### 7.1 문제 설정

```
입력: 제형 특성, 환자 특성
  - 농도, pH, 점도, 삼투압
  - 나이, 체중, 신장, 성별

출력: 약동학 모수
  - Cmax (최대 혈중 농도)
  - Tmax (최대 농도 도달 시간)
  - AUC (혈중 농도-시간 곡선 아래 면적)
  - t½ (반감기)
```

### 7.2 코드 예시

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# 데이터 로드 (예시)
# df = pd.read_csv('pk_data.csv')

# 특성과 타겟 분리
feature_cols = ['conc', 'pH', 'viscosity', 'osmolarity', 
                'age', 'weight', 'height', 'sex']
target_cols = ['Cmax', 'Tmax', 'AUC', 'half_life']

X = df[feature_cols].values
y = df[target_cols].values

# 각 출력에 대해 LightGBM 학습
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

results = {}
for i, target in enumerate(target_cols):
    print(f"\n=== {target} ===")
    
    train_data = lgb.Dataset(X_train, label=y_train[:, i])
    
    # Cross-validation
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        nfold=5,
        callbacks=[lgb.early_stopping(50)]
    )
    
    best_iteration = len(cv_results['valid rmse-mean'])
    best_rmse = cv_results['valid rmse-mean'][-1]
    
    print(f"Best iteration: {best_iteration}")
    print(f"CV RMSE: {best_rmse:.4f}")
    
    results[target] = {
        'best_iter': best_iteration,
        'cv_rmse': best_rmse
    }
```

### 7.3 변수 중요도 분석

```python
# 최종 모델 학습
model = lgb.train(params, train_data, num_boost_round=best_iteration)

# 변수 중요도
importance = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance (Gain)')
plt.title(f'Feature Importance for {target}')
plt.gca().invert_yaxis()
plt.show()
```

---

## 8. 요약

### 8.1 Gradient Boosting

```
핵심: 잔차를 예측하는 트리 순차 추가
장점: 편향 + 분산 감소
단점: 순차 처리, 느림
```

### 8.2 XGBoost

```
개선: 정규화 + 2차 근사 + 병렬화
특징: 정확도 높음, 안정적
```

### 8.3 LightGBM

```
개선: Histogram + Leaf-wise + GOSS + EFB
특징: 빠름, 대용량 데이터에 적합
```

### 8.4 MTGBM

```
확장: 다중 출력 동시 예측
특징: 출력 간 상관관계 활용
```

### 8.5 선택 가이드

```
데이터 작음 + 정확도 중요 → XGBoost
데이터 큼 + 속도 중요 → LightGBM
다중 출력 → MTGBM (또는 Multi-output wrapper)
```

---

## 참고문헌

1. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. _Annals of statistics_, 1189-1232.
    
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _KDD_.
    
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. _NeurIPS_.
    
4. Borchani, H., et al. (2015). A survey on multi-output regression. _Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery_.