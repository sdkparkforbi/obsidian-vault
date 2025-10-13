

## 1. GAIN 소개 및 배경

### 1.1 결측치 문제의 정의

- 실세계 데이터의 불완전성 문제
    - 의료 데이터: 환자별 검사 항목의 차이
    - 시계열 데이터: 센서 오류 및 통신 장애
    - 설문조사: 응답자의 선택적 무응답
    - 금융 데이터: 거래 기록의 부분적 손실

### 1.2 결측치 메커니즘의 분류

- MCAR (Missing Completely At Random)
    - 결측이 완전히 무작위로 발생
    - 관측값과 비관측값이 독립적
- MAR (Missing At Random)
    - 결측이 관측된 변수에 의존
    - 관측값을 조건부로 했을 때 무작위
- MNAR (Missing Not At Random)
    - 결측이 비관측값 자체에 의존
    - 가장 처리하기 어려운 패턴

### 1.3 기존 방법의 한계

- 단순 대체법 (Simple Imputation)
    - 평균/중앙값 대체: 분산 과소추정
    - 전방/후방 대체: 시계열 패턴 왜곡
- 회귀 기반 방법
    - 선형성 가정의 제약
    - 변수 간 복잡한 관계 포착 불가
- Multiple Imputation
    - MAR 가정 필요
    - 계산 비용 높음
    - 반복 대체 과정의 복잡성

### 1.4 GAIN 개발 동기

- GAN의 성공적인 생성 능력 활용
- 결측치를 적대적 학습 문제로 재정의
- Discriminator를 통한 실제값과 생성값의 구분
- Hint mechanism을 통한 학습 안정화

## 2. GAIN 모형 구조

### 2.1 모형 개요

#### 2.1.1 핵심 구성 요소

- Generator ($G$): 결측값 생성 네트워크
- Discriminator ($D$): 실제값과 생성값 판별 네트워크
- Hint mechanism ($H$): 학습 안정화를 위한 부분 정보 제공

#### 2.1.2 표기법 정의

- $\mathbf{X}$: 완전한 데이터 행렬
- $\mathbf{\tilde{X}}$: 결측이 있는 관측 데이터
- $\mathbf{M}$: 마스크 행렬 (1=관측, 0=결측)
- $\mathbf{Z}$: 무작위 노이즈 벡터
- $\mathbf{H}$: 힌트 벡터

### 2.2 수학적 정식화

#### 2.2.1 Generator 함수

- 입력: 불완전 데이터와 노이즈 $$\mathbf{\hat{X}} = G(\mathbf{\tilde{X}}, \mathbf{M}, \mathbf{Z})$$
    
- 최종 대체값 계산 $$\mathbf{X}_{imputed} = \mathbf{M} \odot \mathbf{\tilde{X}} + (1-\mathbf{M}) \odot \mathbf{\hat{X}}$$
    

여기서 $$\odot$$는 element-wise 곱셈

#### 2.2.2 Discriminator 함수

- 입력: 대체된 데이터와 힌트 $$\mathbf{\hat{M}} = D(\mathbf{X}_{imputed}, \mathbf{H})$$
    
- 출력: 각 원소가 실제값일 확률
    

#### 2.2.3 Hint 메커니즘

- 힌트 벡터 생성

$h_{ij} = \begin{cases} m_{ij} & \text{with probability } p \ 0.5 & \text{with probability } 1-p \end{cases}$

- $p$: hint rate (일반적으로 0.9 사용)

### 2.3 손실 함수

#### 2.3.1 Discriminator 손실

$$\mathcal{L}_D = -\mathbb{E}_{\mathbf{X}, \mathbf{M}}[\mathbf{M}^T \log \mathbf{\hat{M}} + (1-\mathbf{M})^T \log(1-\mathbf{\hat{M}})]$$

- 실제값은 1로, 생성값은 0으로 분류하도록 학습

#### 2.3.2 Generator 손실

$$\mathcal{L}_G = -\mathbb{E}_{\mathbf{X}, \mathbf{M}, \mathbf{Z}}[(1-\mathbf{M})^T \log \mathbf{\hat{M}}] + \alpha \cdot \mathcal{L}_{MSE}$$

여기서 재구성 손실: $$\mathcal{L}_{MSE} = ||\mathbf{M} \odot \mathbf{X} - \mathbf{M} \odot \mathbf{\hat{X}}||_2^2$$

- $\alpha$: 재구성 손실 가중치 (하이퍼파라미터)

### 2.4 파라미터 추정

#### 2.4.1 미니맥스 게임

- 최적화 문제 정의 $$\min_G \max_D V(D, G) = \mathcal{L}_D - \mathcal{L}_G$$

#### 2.4.2 교대 최적화 (Alternating Optimization)

- Step 1: Discriminator 업데이트 $$\theta_D^{(t+1)} = \theta_D^{(t)} - \eta_D \nabla_{\theta_D} \mathcal{L}_D$$
    
- Step 2: Generator 업데이트 $$\theta_G^{(t+1)} = \theta_G^{(t)} - \eta_G \nabla_{\theta_G} \mathcal{L}_G$$
    
- $\eta_D, \eta_G$: 학습률
    
- $\theta_D, \theta_G$: 네트워크 파라미터
    

#### 2.4.3 주요 하이퍼파라미터

- hint_rate ($p$): 0.9 권장
    - Discriminator에 제공되는 정보량 조절
    - 높을수록 안정적 학습
- alpha ($\alpha$): 10 권장
    - 재구성 손실의 중요도
    - 높을수록 원본과 유사한 값 생성
- batch_size: 128 권장
    - 메모리와 학습 안정성의 균형
- learning_rate: 0.001 권장
    - Adam optimizer 사용 시 기본값

## 3. 코드 구현

### 3.1 환경 설정

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
```

### 3.2 시뮬레이션 데이터 생성

```python
# Generate regression data
n_samples = 1000
n_features = 10

X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                       n_informative=8, noise=10, random_state=42)

feature_names = [f'Feature_{i+1}' for i in range(n_features)]
df_original = pd.DataFrame(X, columns=feature_names)
df_original['Target'] = y
```

### 3.3 결측치 생성 함수

```python
def create_missing_data(data, missing_rate=0.3, pattern='random'):
    """Create various missing patterns"""
    data_with_missing = data.copy()
    n_samples, n_features = data.shape
    
    if pattern == 'random':
        # MCAR pattern
        mask = np.random.random((n_samples, n_features)) < missing_rate
        data_with_missing[mask] = np.nan
        
    return data_with_missing

X_array = df_original[feature_names].values
X_missing = create_missing_data(X_array, missing_rate=0.3)
M = (~np.isnan(X_missing)).astype(np.float32)  # Mask matrix
```

### 3.4 GAIN 모델 구현

```python
class SimpleGAIN(tf.keras.Model):
    """Simplified GAIN model for education"""
    
    def __init__(self, data_dim, hidden_dim=64):
        super(SimpleGAIN, self).__init__()
        self.data_dim = data_dim
        
        # Generator network
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(data_dim, activation='linear')
        ])
        
        # Discriminator network
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(data_dim, activation='sigmoid')
        ])
    
    def generate(self, X, M):
        inputs = tf.concat([X, M], axis=1)
        return self.generator(inputs)
    
    def discriminate(self, X, hint):
        inputs = tf.concat([X, hint], axis=1)
        return self.discriminator(inputs)
```

### 3.5 데이터 전처리

```python
# Normalization using observed values only
scaler = StandardScaler()
X_observed = X_missing[~np.isnan(X_missing)]
scaler.fit(X_observed.reshape(-1, 1))

# Apply normalization
X_normalized = X_missing.copy()
for i in range(n_features):
    col_data = X_missing[:, i]
    observed_mask = ~np.isnan(col_data)
    if observed_mask.any():
        X_normalized[observed_mask, i] = scaler.transform(
            col_data[observed_mask].reshape(-1, 1)).flatten()
        X_normalized[~observed_mask, i] = 0
```

### 3.6 GAIN 학습 루프

```python
# Initialize model and optimizers
gain = SimpleGAIN(n_features)
optimizer_G = tf.keras.optimizers.Adam(0.001)
optimizer_D = tf.keras.optimizers.Adam(0.001)

# Training parameters
n_epochs = 100
batch_size = 128
alpha = 10
hint_rate = 0.9

history = {'d_loss': [], 'g_loss': []}

for epoch in range(n_epochs):
    indices = np.random.permutation(n_samples)
    n_batches = n_samples // batch_size
    
    epoch_d_losses = []
    epoch_g_losses = []
    
    for batch_idx in range(n_batches):
        batch_indices = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]
        X_batch = tf.constant(X_normalized[batch_indices], dtype=tf.float32)
        M_batch = tf.constant(M[batch_indices], dtype=tf.float32)
        
        # Generate hint
        hint = M_batch * np.random.binomial(1, hint_rate, M_batch.shape).astype(np.float32)
        hint = hint + 0.5 * (1 - hint)
        
        # Train Discriminator
        with tf.GradientTape() as tape:
            G_sample = gain.generate(X_batch, M_batch)
            X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
            D_prob = gain.discriminate(X_hat, hint)
            
            D_loss = -tf.reduce_mean(M_batch * tf.math.log(D_prob + 1e-8) + 
                                    (1-M_batch) * tf.math.log(1 - D_prob + 1e-8))
        
        gradients = tape.gradient(D_loss, gain.discriminator.trainable_variables)
        optimizer_D.apply_gradients(zip(gradients, gain.discriminator.trainable_variables))
        
        # Train Generator
        with tf.GradientTape() as tape:
            G_sample = gain.generate(X_batch, M_batch)
            X_hat = M_batch * X_batch + (1 - M_batch) * G_sample
            D_prob = gain.discriminate(X_hat, hint)
            
            G_loss_adv = -tf.reduce_mean((1-M_batch) * tf.math.log(D_prob + 1e-8))
            X_true = tf.constant(X_array[batch_indices], dtype=tf.float32)
            X_true_norm = (X_true - scaler.mean_[0]) / scaler.scale_[0]
            G_loss_rec = tf.reduce_mean((1-M_batch) * tf.square(G_sample - X_true_norm))
            G_loss = G_loss_adv + alpha * G_loss_rec
        
        gradients = tape.gradient(G_loss, gain.generator.trainable_variables)
        optimizer_G.apply_gradients(zip(gradients, gain.generator.trainable_variables))
        
        epoch_d_losses.append(D_loss.numpy())
        epoch_g_losses.append(G_loss.numpy())
    
    history['d_loss'].append(np.mean(epoch_d_losses))
    history['g_loss'].append(np.mean(epoch_g_losses))
```

### 3.7 대체값 생성 및 평가

```python
# Generate imputed values
X_normalized_tf = tf.constant(X_normalized, dtype=tf.float32)
M_tf = tf.constant(M, dtype=tf.float32)

G_sample = gain.generate(X_normalized_tf, M_tf)
X_imputed_normalized = M * X_normalized + (1 - M) * G_sample.numpy()

# Denormalize
X_imputed = X_imputed_normalized * scaler.scale_[0] + scaler.mean_[0]

# Evaluation
missing_mask = (M == 0)
true_values = X_array[missing_mask]
imputed_values = X_imputed[missing_mask]

rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
mae = np.mean(np.abs(true_values - imputed_values))
correlation = np.corrcoef(true_values, imputed_values)[0, 1]
```

### 3.8 결과 시각화

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True vs Imputed scatter plot
axes[0].scatter(true_values, imputed_values, alpha=0.5, s=1)
axes[0].plot([true_values.min(), true_values.max()], 
            [true_values.min(), true_values.max()], 'r--')
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Imputed Values')
axes[0].set_title(f'True vs Imputed (r={correlation:.3f})')

# Distribution comparison
axes[1].hist(true_values, bins=30, alpha=0.5, label='True', density=True)
axes[1].hist(imputed_values, bins=30, alpha=0.5, label='Imputed', density=True)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution Comparison')
axes[1].legend()

# Training loss curves
axes[2].plot(history['d_loss'], label='Discriminator')
axes[2].plot(history['g_loss'], label='Generator')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('Training Loss')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## 4. 참고문헌

### 4.1 핵심 논문

Yoon, J., Jordon, J., & Schaar, M. (2018). GAIN: Missing data imputation using generative adversarial nets. _International Conference on Machine Learning_, 5689-5698. https://doi.org/10.48550/arXiv.1806.02920

### 4.2 관련 연구

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. _Advances in Neural Information Processing Systems_, 27. https://doi.org/10.48550/arXiv.1406.2661

Little, R. J., & Rubin, D. B. (2019). _Statistical analysis with missing data_ (3rd ed.). John Wiley & Sons. https://doi.org/10.1002/9781119482260

Mattei, P. A., & Frellsen, J. (2019). MIWAE: Deep generative modelling and imputation of incomplete data sets. _International Conference on Machine Learning_, 4413-4423. https://doi.org/10.48550/arXiv.1812.02633

Nazabal, A., Olmos, P. M., Ghahramani, Z., & Valera, I. (2020). Handling incomplete heterogeneous data using VAEs. _Pattern Recognition_, 107, 107501. https://doi.org/10.1016/j.patcog.2020.107501

Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. _Bioinformatics_, 28(1), 112-118. https://doi.org/10.1093/bioinformatics/btr597

Van Buuren, S. (2018). _Flexible imputation of missing data_ (2nd ed.). CRC Press. https://doi.org/10.1201/9780429492259

### 4.3 응용 분야별 연구

Luo, Y., Zhang, Y., Cai, X., & Yuan, X. (2019). E2GAN: End-to-end generative adversarial network for multivariate time series imputation. _Proceedings of the 28th International Joint Conference on Artificial Intelligence_, 3094-3100. https://doi.org/10.24963/ijcai.2019/429

Camino, R. D., Hammerschmidt, C. A., & State, R. (2019). Improving missing data imputation with deep generative models. _arXiv preprint arXiv:1902.10666_. https://doi.org/10.48550/arXiv.1902.10666

Zhang, H., Xie, P., & Xing, E. (2018). Missing value imputation based on deep generative models. _arXiv preprint arXiv:1808.01684_. https://doi.org/10.48550/arXiv.1808.01684