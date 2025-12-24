# GAIN (Generative Adversarial Imputation Networks) 튜토리얼

---

## 1. 개요 (Overview)

### 1.1 프로그램의 목적
- 약동학(Pharmacokinetics, PK) 데이터에서 결측치(Missing Values)를 대체(Imputation)하는 딥러닝 기반 방법
- Iwata et al. (2022) 논문 데이터를 활용한 GAIN vs 기존 방법(Iwata) 성능 비교

### 1.2 주요 구성 요소
- 데이터 로딩 및 전처리 (Iwata et al. 데이터)
- GAIN 모델 구현 (TensorFlow/Keras)
- 학습 및 검증
- LightGBM을 활용한 다운스트림 예측 성능 비교

---

## 2. 배경 지식 (Background)

### 2.1 결측치 대체(Imputation)란?

데이터셋에서 일부 값이 관측되지 않은 경우(NaN, NA), 이를 적절한 값으로 채워 넣는 과정

**전통적인 방법들:**
- Mean/Median Imputation: 해당 변수의 평균/중앙값으로 대체
- KNN Imputation: K-최근접 이웃의 값을 참조하여 대체
- Multiple Imputation: 여러 번 대체하여 불확실성을 반영
- Matrix Factorization: 행렬 분해를 통한 대체

**딥러닝 기반 방법:**
- Autoencoder 기반 대체
- GAIN (본 프로그램): GAN 구조를 결측치 대체에 응용

### 2.2 GAIN의 기본 아이디어

GAIN은 GAN(Generative Adversarial Network)의 적대적 학습 구조를 결측치 대체 문제에 응용한 방법이다.

**GAN의 핵심 (간략 요약):**
- Generator(생성자)와 Discriminator(판별자)가 서로 경쟁하며 학습
- Generator는 그럴듯한 데이터를 생성하고, Discriminator는 진위를 판별
- 이 경쟁을 통해 Generator가 점점 더 실제와 유사한 데이터를 생성

**GAIN의 핵심 변형:**
- Generator: 결측된 위치의 값을 생성
- Discriminator: 각 위치가 원본인지 대체값인지 판별 (위치별 판별)
- Hint Mechanism: Discriminator에게 부분적 정보를 제공하여 학습 안정화

---

## 3. GAIN 모델 구조

### 3.1 GAN과 GAIN의 핵심 차이

| 구분                   | GAN            | GAIN                   |
| -------------------- | -------------- | ---------------------- |
| **목적**               | 새로운 데이터 생성     | 결측치 대체                 |
| **Generator 입력**     | 랜덤 노이즈 $z$     | 불완전한 데이터 $X$ + 마스크 $M$ |
| **Discriminator 출력** | 스칼라 (샘플 전체 판별) | 벡터 (각 위치별 판별)          |
| **핵심 질문**            | "이 샘플이 진짜인가?"  | "각 위치가 원본인가?"          |

### 3.2 GAIN 구성 요소

**1. Mask Matrix (M):** 관측 여부를 나타내는 이진 행렬
- $M_{ij} = 1$: 관측됨
- $M_{ij} = 0$: 결측됨

**2. Hint Matrix (H):** Discriminator에게 부분적 정보를 제공
- 일부 위치에서 실제 마스크 정보를 알려줌
- Discriminator가 너무 쉽게 학습하는 것을 방지

**3. Generator:** 결측 위치의 값을 생성
```
입력: [X, M] (데이터 + 마스크 연결)
출력: 모든 위치의 값 (결측 위치만 실제 사용)
```

**4. Discriminator:** 각 위치가 원본인지 대체값인지 판별
```
입력: [X̂, H] (대체된 데이터 + 힌트)
출력: 각 위치가 원본일 확률 (벡터)
```

### 3.3 GAIN 처리 흐름

```
입력 데이터:
    X = [4, ?, 2, ?, 5]  (불완전 데이터)
    M = [1, 0, 1, 0, 1]  (마스크: 1=관측, 0=결측)
              │
              ▼
         ┌─────────┐
         │Generator│
         └────┬────┘
              │
              ▼
    G(X,M) = [4.1, 3.2, 2.0, 4.5, 5.1]  (모든 위치 생성)
              │
              ▼
    X̂ = M⊙X + (1-M)⊙G(X,M)
       = [4, 3.2, 2, 4.5, 5]  (관측 위치는 원본 유지, 결측만 대체)
              │
    H = [1, 0.5, 0.5, 0.5, 1]  (힌트)
              │
              ▼
         ┌───────────────┐
         │ Discriminator │
         └───────┬───────┘
                 │
                 ▼
    D(X̂,H) = [0.9, 0.6, 0.8, 0.5, 0.95]  (각 위치가 원본일 확률)
```

### 3.4 목적 함수

**최적화 문제:**

$$\min_G \max_D V(D, G)$$

where

$$V(D, G) = \mathbb{E}_{\hat{X}, M, H}\left[ M^T \log D(\hat{X}, H) + (1-M)^T \log(1 - D(\hat{X}, H)) \right]$$

**각 항의 의미:**
- $M^T \log D(\hat{X}, H)$: 관측 위치에서 "원본"이라고 판단하도록
- $(1-M)^T \log(1 - D(\hat{X}, H))$: 결측 위치에서 "대체값"이라고 판단하도록

**Generator 손실 (추가):**

$$\mathcal{L}_G = \mathcal{L}_{adversarial} + \alpha \cdot \mathcal{L}_{reconstruction}$$

- Adversarial Loss: Discriminator를 속이는 방향
- Reconstruction Loss: 실제값(Ground Truth)과의 차이 최소화
- $\alpha = 10$: 가중치 (본 프로그램 설정)

### 3.5 Hint Mechanism 상세

**Hint Rate ($\rho$):** 본 프로그램에서 0.9 사용

$$H_{ij} = M_{ij} \cdot B_{ij} + 0.5 \cdot (1 - M_{ij} \cdot B_{ij})$$

여기서 $B_{ij} \sim \text{Bernoulli}(\rho)$

**의미:**
- $H_{ij} = 1$: 이 위치는 확실히 원본 (Discriminator에게 힌트 제공)
- $H_{ij} = 0.5$: 불확실 (Discriminator가 추론해야 함)

**왜 필요한가?**
- 힌트 없이 학습하면 Discriminator가 "결측 패턴"만 보고 쉽게 구분
- Generator가 제대로 학습되지 않음
- 힌트를 통해 Discriminator의 난이도를 조절하여 균형 잡힌 학습 유도

### 3.6 본 프로그램의 네트워크 구조

**Generator:**
```
입력: [X_scaled, M] (데이터와 마스크의 연결)
    ↓
Dense(128, ReLU)
    ↓
BatchNormalization
    ↓
Dropout(0.2)
    ↓
Dense(64, ReLU)
    ↓
BatchNormalization
    ↓
Dense(data_dim, Linear)  # 출력: 대체된 값
```

**Discriminator:**
```
입력: [X_hat, Hint] (대체된 데이터 + 힌트)
    ↓
Dense(128, ReLU)
    ↓
BatchNormalization
    ↓
Dropout(0.2)
    ↓
Dense(64, ReLU)
    ↓
BatchNormalization
    ↓
Dense(data_dim, Sigmoid)  # 출력: 각 위치가 원본일 확률
```

---

## 4. 프로그램 워크플로우

### 4.1 데이터 로딩 (Iwata et al. 데이터)

**데이터 출처:** ACS Publications (Journal of Chemical Information and Modeling)

**변수 목록 (16개 수치형):**
- PK 파라미터: human_CL_mL_min_kg, human_VDss_L_kg (예측 타겟)
- 종간 PK: rat_CL, rat_VDss, dog_CL, dog_VDss, monkey_CL, monkey_VDss
- 단백질 결합: human_fup, rat_fup, dog_fup, monkey_fup
- 물리화학적: pKa_Acid, pKa_base, water_solubility, Caco_2

**데이터 구조:**
- Sheet 1: Raw data (770 compounds, 결측 多)
- Sheet 2-5: Iwata 방법으로 대체된 데이터

### 4.2 전처리 파이프라인

**Step 1: 극단값 필터링**
```python
condition = (CL >= 20) | (VDss >= 5)
# 조건 만족하는 샘플 제거
```
- 목적: 비정상적으로 높은 청소율/분포용적 제거
- 722 → 581 samples

**Step 2: Winsorization (이상치 조정)**
```python
lower_bound = np.percentile(data, 1)
upper_bound = np.percentile(data, 99)
data_clipped = np.clip(data, lower_bound, upper_bound)
```
- 1st-99th 백분위수로 극단값 제한
- 분포의 꼬리 부분 영향 감소

### 4.3 Train/Valid/Test 분할

**동질성 보장 분할 (Homogeneous Splits):**
```python
for seed in range(100):
    # 분할 수행
    # Kruskal-Wallis 검정으로 동질성 평가
    # 가장 높은 p-value를 가진 분할 선택
```

- 비율: 60% / 20% / 20%
- Kruskal-Wallis H-test로 Train/Valid/Test 간 분포 동질성 확인

### 4.4 인위적 결측 추가 (Artificial Missing)

**목적:** 모델 학습 및 평가를 위한 Ground Truth 확보

```python
def add_artificial_missing(X, missing_rate=0.1):
    # 현재 관측된 값 중 10%를 인위적으로 결측 처리
    for each feature:
        observed_indices = where(~isnan(X[:, feature]))
        n_to_remove = int(len(observed_indices) * missing_rate)
        randomly_select_and_set_to_nan(n_to_remove)
    return X_with_artificial_missing
```

**결과:**
- Original Missing: 모델이 예측해야 할 진짜 결측
- Artificial Missing: 평가용 (Ground Truth 존재)

### 4.5 GAIN 학습

**하이퍼파라미터:**
- Epochs: 최대 1000 (Early Stopping)
- Batch Size: 32
- Learning Rate: 0.0001 (Adam optimizer)
- Hint Rate: 0.9
- Alpha (Reconstruction weight): 10
- Early Stopping Patience: 8 (Validation RMSE 기준)

---

## 5. 평가 지표

### 5.1 Imputation 성능 평가

**RMSE (Root Mean Square Error):**
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**MAE (Mean Absolute Error):**
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Correlation:**
$$r = \frac{\sum(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum(y_i - \bar{y})^2 \sum(\hat{y}_i - \bar{\hat{y}})^2}}$$

**GMFE (Geometric Mean Fold Error):**
$$\text{GMFE} = 10^{\frac{1}{n}\sum|\log_{10}(y_i / \hat{y}_i)|}$$
- 약동학에서 자주 사용되는 지표
- 1에 가까울수록 좋음

### 5.2 다운스트림 예측 성능

대체된 데이터로 human_CL, human_VDss 예측 모델(LightGBM) 학습

**평가 지표:**
- $R^2$ (결정계수)
- RMSE
- MAE

---

## 6. 실행 결과

### 6.1 GAIN 학습 성능

| 항목                    | 값                    |
| --------------------- | -------------------- |
| 총 학습 Epoch            | 201 (Early Stopping) |
| 최적 Validation RMSE    | 0.2407 (Epoch 50)    |
| 최종 Generator Loss     | 0.4997               |
| 최종 Discriminator Loss | 0.6011               |

### 6.2 Imputation 품질

**전체 통계:** Mean correlation = **0.642 ± 0.187**

**상위 Feature 성능:**

| Rank | Feature | r | R² | n |
|------|---------|---|-----|---|
| 1 | water_solubility | 0.90 | 0.63 | 27 |
| 2 | Caco_2 | 0.84 | 0.56 | 23 |
| 3 | dog_VDss_L_kg | 0.84 | 0.43 | 22 |
| 8 | human_CL_mL_min_kg | 0.69 | 0.43 | 56 |
| 9 | human_VDss_L_kg | 0.69 | 0.47 | 56 |

**분포 보존:** Kolmogorov-Smirnov 검정 결과 84.4% feature에서 유의한 분포 차이 없음

### 6.3 GAIN vs Iwata 비교 (다운스트림 예측)

| Target | Metric | GAIN | Iwata | Winner | Improvement |
|--------|--------|------|-------|--------|-------------|
| **CL** | R² | 0.41 | 0.15 | GAIN | **+170.9%** |
| **CL** | RMSE | 3.60 | 4.32 | GAIN | -16.6% |
| **CL** | MAE | 2.61 | 3.35 | GAIN | -22.0% |
| **VDss** | R² | 0.50 | 0.44 | GAIN | +13.2% |
| **VDss** | RMSE | 0.72 | 0.76 | GAIN | -5.4% |
| **VDss** | MAE | 0.52 | 0.55 | GAIN | -5.9% |

**최종 결과: GAIN 6승 / Iwata 0승**

---

## 7. 핵심 코드 분석

### 7.1 GAIN 클래스 정의

```python
class GAIN(tf.keras.Model):
    def __init__(self, data_dim, hint_rate=0.9, alpha=10):
        super(GAIN, self).__init__()
        self.data_dim = data_dim
        self.hint_rate = hint_rate
        self.alpha = alpha  # Reconstruction loss weight
        
        # Generator: 결측값 생성
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(data_dim, activation='linear')
        ])
        
        # Discriminator: 원본 vs 대체 구분
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(data_dim, activation='sigmoid')
        ])
```

### 7.2 Imputation 수행

```python
def generate(self, X, mask):
    """Generator로 값 생성"""
    G_input = tf.concat([X, mask], axis=1)  # 데이터 + 마스크 연결
    return self.generator(G_input)

# 최종 대체 결과
G_sample = gain.generate(X_incomplete, M)
X_imputed = M * X_incomplete + (1 - M) * G_sample
# 관측된 위치: 원본 유지
# 결측된 위치: Generator 출력으로 대체
```

### 7.3 학습 단계

```python
# Discriminator 학습
with tf.GradientTape() as tape_D:
    G_sample = gain.generate(X_batch, M_batch)
    X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
    D_prob = gain.discriminate(X_hat, hint)
    
    # 위치별 Binary Cross-Entropy Loss
    D_loss = -tf.reduce_mean(
        M_batch * tf.math.log(D_prob + 1e-8) +
        (1 - M_batch) * tf.math.log(1 - D_prob + 1e-8)
    )

# Generator 학습
with tf.GradientTape() as tape_G:
    G_sample = gain.generate(X_batch, M_batch)
    X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
    D_fake = gain.discriminate(X_hat, hint)
    
    # Adversarial Loss: Discriminator를 속이는 방향
    G_loss_adv = -tf.reduce_mean(
        (1 - M_batch) * tf.math.log(D_fake + 1e-8)
    )
    
    # Reconstruction Loss: Ground Truth와의 차이
    G_loss_recon = tf.reduce_mean(
        (1 - M_batch) * tf.square(G_sample - X_complete)
    )
    
    G_loss = G_loss_adv + alpha * G_loss_recon
```

---

## 8. 약동학 배경 지식

### 8.1 주요 PK 파라미터

**Clearance (CL, 청소율):**
- 단위 시간당 약물이 제거되는 혈장의 부피
- 단위: mL/min/kg
- 의미: 약물 제거 속도

**Volume of Distribution at Steady State (VDss, 정상 상태 분포용적):**
- 약물이 체내에 균일하게 분포한다고 가정했을 때의 가상 부피
- 단위: L/kg
- 의미: 약물의 조직 분포 정도

**Fraction Unbound in Plasma (fup, 혈장 비결합률):**
- 혈장 단백질에 결합하지 않은 자유 약물 비율
- 범위: 0~1

### 8.2 종간 스케일링 (Allometric Scaling)

동물 데이터로부터 인체 PK 예측:

본 데이터셋 활용:
- Rat, Dog, Monkey의 PK 데이터
- 물리화학적 특성 (pKa, 용해도, 투과도)
- → Human CL, VDss 예측

---

## 9. 참고 문헌

1. **GAIN 원논문:** Yoon, J., Jordon, J., & Van Der Schaar, M. (2018). GAIN: Missing data imputation using generative adversarial nets. *ICML*.

2. **Iwata et al. 데이터:** Iwata, H., et al. (2022). Predicting Total Drug Clearance and Volumes of Distribution... *J. Chem. Inf. Model.* DOI: 10.1021/acs.jcim.2c00318

---

## 부록: 용어 정리

| 용어 | 설명 |
|------|------|
| Imputation | 결측치를 추정값으로 대체하는 과정 |
| GAIN | GAN 구조를 결측치 대체에 응용한 방법 |
| Mask Matrix | 관측 여부를 나타내는 이진 행렬 |
| Hint Matrix | Discriminator에게 부분 정보를 제공하는 행렬 |
| Reconstruction Loss | 생성값과 실제값 간의 차이를 측정하는 손실 |
| Winsorization | 극단값을 특정 백분위수로 제한하는 방법 |
| Early Stopping | 검증 성능이 개선되지 않으면 학습을 조기 종료 |
| Downstream Task | 대체된 데이터를 활용한 후속 예측 작업 |
