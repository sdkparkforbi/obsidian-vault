
## 1. BASS 모형 개요

### 1) 모형의 배경

- Frank Bass(1969)가 개발한 신제품 확산 모형
- 혁신의 채택과 확산 과정을 수학적으로 모델링
- 마케팅, 기술 확산, 전염병 확산 등에 광범위하게 활용

### 2) 기본 가정

- 잠재 채택자의 총 수는 고정(m)
- 채택은 혁신효과와 모방효과에 의해 발생
- 한번 채택하면 재구매나 포기는 없음

## 2. 미분방정식 유도

### 1) Hazard Rate 개념

**Hazard Rate (위험률)**

- 시점 t까지 채택하지 않은 개체가 시점 t에서 채택할 조건부 확률
- 정의:

$$h(t) = \frac{f(t)}{1-F(t)}$$

여기서,

- f(t): 채택 시점의 확률밀도함수
- F(t): 누적분포함수 (시점 t까지의 누적 채택 비율)
- 1-F(t): 생존함수 (아직 채택하지 않은 비율)

### 2) BASS 모형의 Hazard Rate

BASS 모형의 핵심 가정:

$$h(t) = p + q \cdot F(t)$$

여기서,

- p: 혁신계수 (innovation coefficient) - 외부 영향
- q: 모방계수 (imitation coefficient) - 내부 영향
- F(t): 시점 t까지의 누적 채택 비율

### 3) 미분방정식 유도

Hazard rate 정의로부터:

$$\frac{f(t)}{1-F(t)} = p + q \cdot F(t)$$

양변을 정리하면:

$$f(t) = (p + q \cdot F(t)) \cdot (1 - F(t))$$

f(t) = dF(t)/dt이므로:

$$\frac{dF(t)}{dt} = (p + q \cdot F(t)) \cdot (1 - F(t))$$

누적 채택자 수 N(t) = m·F(t)로 표현하면:

$$\frac{dN(t)}{dt} = \frac{p + q \cdot \frac{N(t)}{m}}{1} \cdot (m - N(t))$$

최종 BASS 미분방정식:

$$\frac{dN(t)}{dt} = p(m - N(t)) + \frac{q}{m} \cdot N(t) \cdot (m - N(t))$$

## 3. 모형 파라미터

### 1) 파라미터 정의

**세 가지 핵심 파라미터**

- m: 잠재 시장 규모 (최대 채택자 수)
- p: 혁신계수 (0 < p < 1)
    - 광고, 외부 정보에 의한 채택
    - 일반적으로 0.001 ~ 0.05 범위
- q: 모방계수 (0 < q < 1)
    - 기존 채택자의 구전효과
    - 일반적으로 0.1 ~ 0.5 범위

### 2) 해석적 해

**누적 채택 함수**

BASS 모형의 해석적 해:

$N(t) = m \cdot \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} \cdot e^{-(p+q)t}}$

또는 누적 분포 형태로:

$F(t) = \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} \cdot e^{-(p+q)t}}$

**확률밀도함수 (신규 채택률)**

F(t)를 미분하여 얻는 확률밀도함수:

$f(t) = \frac{dF(t)}{dt} = \frac{(p+q)^2 \cdot e^{-(p+q)t}}{p \cdot \left(1 + \frac{q}{p} \cdot e^{-(p+q)t}\right)^2}$

신규 채택자 수로 표현하면:

$n(t) = \frac{dN(t)}{dt} = m \cdot f(t) = \frac{m \cdot (p+q)^2 \cdot e^{-(p+q)t}}{p \cdot \left(1 + \frac{q}{p} \cdot e^{-(p+q)t}\right)^2}$

다른 유용한 형태로 정리하면:

$n(t) = \frac{m \cdot (p + q \cdot F(t))^2}{p + q}$

### 3) 특성 시점

**채택 정점 시점 (Peak time)**

$$t^* = \frac{1}{p+q} \cdot \ln\left(\frac{q}{p}\right)$$

**정점에서의 채택률**

$$\frac{dN(t^*)}{dt} = \frac{m(p+q)^2}{4q}$$

## 4. 파라미터 추정 방법

### 1) 추정에 필요한 데이터

- 시계열 판매/채택 데이터
- 기간별 신규 채택자 수 또는 누적 채택자 수
- 최소 3개 이상의 시점 데이터 필요

### 2) 주요 추정 방법

**비선형 최소제곱법 (NLS)**

- 가장 널리 사용되는 방법
- 실제 데이터와 모형 예측값의 오차 제곱합 최소화
- 목적함수:

$$SSE = \sum_{t=1}^{T} (N_t^{observed} - N_t^{predicted})^2$$

**최대우도추정법 (MLE)**

- 우도함수를 최대화하는 파라미터 찾기
- 통계적 성질이 우수
- 우도함수:

$$L(p,q,m|data) = \prod_{t=1}^{T} f(t|p,q,m)$$

**선형 근사법**

- 이산시간 버전으로 변환 후 OLS 적용
- 초기 추정값 구하는데 유용
- 회귀식:

$$n_t = a + b \cdot N_{t-1} + c \cdot N_{t-1}^2$$

## 5. Python 구현 (Google Colab)


https://colab.research.google.com/drive/11W79ZO8OuBfSO8XXDPE0QqJJCQ2YLtu5?usp=sharing

### 1) 필요 라이브러리 및 기본 설정

```python
"""
BASS 확산 모형 구현 및 파라미터 추정
Google Colab에서 실행 가능한 완전한 예제
"""

# Google Colab 환경
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
!pip install -q koreanize_matplotlib
import koreanize_matplotlib
import matplotlib.font_manager as fm

# 현재 설정된 폰트 이름 확인
font_name = plt.rcParams['font.family'][0]
print(f"현재 폰트 이름: {font_name}")

# 해당 폰트의 실제 파일 경로 찾기
font_path = fm.findfont(font_name)
print(f"폰트 파일 경로: {font_path}")

# 해당 폰트의 속성
font_prop = fm.FontProperties(fname=font_path)
print(f"폰트 속성: {font_prop}")
 
print("환경 설정 완료")
```

### 2) BASS 모형 함수 정의

```python
def bass_model(t, m, p, q):
    """
    BASS 모형의 누적 채택자 수 계산
    
    Parameters:
    -----------
    t : array-like, 시간
    m : float, 잠재 시장 규모
    p : float, 혁신계수
    q : float, 모방계수
    
    Returns:
    --------
    N(t) : 누적 채택자 수
    """
    exp_term = np.exp(-(p + q) * t)
    N_t = m * (1 - exp_term) / (1 + (q/p) * exp_term)
    return N_t

def bass_density(t, m, p, q):
    """
    BASS 모형의 확률밀도함수 (신규 채택자 수)
    n(t) = m * f(t)
    
    Parameters:
    -----------
    t : array-like, 시간
    m : float, 잠재 시장 규모
    p : float, 혁신계수
    q : float, 모방계수
    
    Returns:
    --------
    n(t) : 신규 채택자 수
    """
    exp_term = np.exp(-(p + q) * t)
    numerator = m * (p + q)**2 * exp_term
    denominator = p * (1 + (q/p) * exp_term)**2
    n_t = numerator / denominator
    return n_t

def bass_differential(N, t, m, p, q):
    """
    BASS 모형의 미분방정식
    dN/dt = p(m-N) + (q/m)*N*(m-N)
    """
    dNdt = p * (m - N) + (q/m) * N * (m - N)
    return dNdt

def bass_hazard_rate(t, p, q):
    """
    BASS 모형의 Hazard Rate 계산
    h(t) = p + q*F(t)
    """
    exp_term = np.exp(-(p + q) * t)
    F_t = (1 - exp_term) / (1 + (q/p) * exp_term)
    h_t = p + q * F_t
    return h_t
```

### 3) 샘플 데이터 생성 및 파라미터 추정

```python
# 실제 파라미터 설정 (스마트폰 확산 사례를 모방)
true_m = 10000  # 잠재 시장 규모 (만 명)
true_p = 0.03   # 혁신계수
true_q = 0.38   # 모방계수

# 시간 범위 설정 (20개 분기)
t_data = np.arange(0, 20)

# 실제 데이터 생성 (노이즈 추가)
np.random.seed(42)
true_adoption = bass_model(t_data, true_m, true_p, true_q)
noise = np.random.normal(0, 50, len(t_data))
observed_adoption = np.maximum(0, true_adoption + noise)

# 파라미터 추정을 위한 목적함수
def objective_function(params, t, y_observed):
    """최소화할 목적함수 (잔차 제곱합)"""
    m, p, q = params
    y_pred = bass_model(t, m, p, q)
    sse = np.sum((y_observed - y_pred)**2)
    return sse

# 초기값 설정 (매우 중요!)
initial_guess = [max(observed_adoption) * 1.5, 0.01, 0.3]

# 제약 조건 설정
bounds = [(max(observed_adoption), max(observed_adoption) * 3),  # m의 범위
          (0.001, 0.1),  # p의 범위  
          (0.01, 0.9)]   # q의 범위

# 최적화 실행
result = minimize(objective_function, initial_guess, 
                 args=(t_data, observed_adoption),
                 bounds=bounds, 
                 method='L-BFGS-B')

estimated_m, estimated_p, estimated_q = result.x

print("="*60)
print("파라미터 추정 결과")
print("="*60)
print(f"추정된 파라미터:")
print(f"  - 잠재 시장 규모 (m): {estimated_m:,.0f}")
print(f"  - 혁신계수 (p): {estimated_p:.4f}")
print(f"  - 모방계수 (q): {estimated_q:.4f}")
```

### 4) 모형 적합도 평가 및 시각화

```python
# 예측값 계산
predicted_adoption = bass_model(t_data, estimated_m, estimated_p, estimated_q)

# R-squared 계산
ss_res = np.sum((observed_adoption - predicted_adoption)**2)
ss_tot = np.sum((observed_adoption - np.mean(observed_adoption))**2)
r_squared = 1 - (ss_res / ss_tot)

# RMSE 계산
rmse = np.sqrt(np.mean((observed_adoption - predicted_adoption)**2))

print("\n" + "="*60)
print("모형 적합도 평가")
print("="*60)
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {rmse:.2f}")

# 시각화 (Seaborn 스타일 적용)
sns.set_style("whitegrid")
sns.set(font=font_name)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) 누적 채택자 수
t_extended = np.linspace(0, 30, 100)
predicted_extended = bass_model(t_extended, estimated_m, estimated_p, estimated_q)

ax1 = axes[0, 0]
ax1.scatter(t_data, observed_adoption, color='blue', alpha=0.6, s=50, label='관측 데이터')
ax1.plot(t_extended, predicted_extended, 'r-', linewidth=2, label='BASS 모형 적합')
ax1.axhline(y=estimated_m, color='green', linestyle='--', alpha=0.5, label=f'm={estimated_m:.0f}')
ax1.set_xlabel('시간 (분기)')
ax1.set_ylabel('누적 채택자 수')
ax1.set_title('BASS 모형 - 누적 채택자 수')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2) 신규 채택자 수 (채택률) - 해석해 사용
ax2 = axes[0, 1]
# 해석해로 계산한 신규 채택자 수
adoption_rate = bass_density(t_extended, estimated_m, estimated_p, estimated_q)
ax2.plot(t_extended, adoption_rate, 'r-', linewidth=2, label='f(t) 해석해')
ax2.fill_between(t_extended, 0, adoption_rate, alpha=0.3, color='red')

# 실제 데이터의 차분값 비교
actual_new = np.diff(np.concatenate(([0], observed_adoption)))
t_actual = t_data[:-1] + 0.5  # 중점
ax2.scatter(t_actual, actual_new[:-1], color='blue', alpha=0.6, s=30, label='실제 신규 채택')

ax2.set_xlabel('시간 (분기)')
ax2.set_ylabel('신규 채택자 수')
ax2.set_title('BASS 모형 - 신규 채택자 수 n(t) = m·f(t)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3) Hazard Rate
ax3 = axes[1, 0]
hazard = bass_hazard_rate(t_extended, estimated_p, estimated_q)
ax3.plot(t_extended, hazard, 'b-', linewidth=2)
ax3.axhline(y=estimated_p, color='red', linestyle='--', alpha=0.5, label=f'p = {estimated_p:.4f}')
ax3.axhline(y=estimated_p + estimated_q, color='green', linestyle='--', alpha=0.5, label=f'p+q = {estimated_p + estimated_q:.4f}')
ax3.set_xlabel('시간 (분기)')
ax3.set_ylabel('Hazard Rate')
ax3.set_title('BASS 모형 - Hazard Rate: h(t) = p + q*F(t)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4) 혁신 vs 모방 효과
innovation_effect = []
imitation_effect = []

for t in t_extended:
    N_t = bass_model(t, estimated_m, estimated_p, estimated_q)
    innovation = estimated_p * (estimated_m - N_t)
    imitation = (estimated_q/estimated_m) * N_t * (estimated_m - N_t)
    innovation_effect.append(innovation)
    imitation_effect.append(imitation)

ax4 = axes[1, 1]
ax4.plot(t_extended, innovation_effect, 'b-', linewidth=2, label='혁신 효과')
ax4.plot(t_extended, imitation_effect, 'r-', linewidth=2, label='모방 효과')
ax4.plot(t_extended, np.array(innovation_effect) + np.array(imitation_effect), 
         'g--', linewidth=2, alpha=0.7, label='총 효과')
ax4.set_xlabel('시간 (분기)')
ax4.set_ylabel('채택 기여도')
ax4.set_title('혁신 효과 vs 모방 효과')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('BASS 확산 모형 분석', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```

### 5) 주요 특성값 계산

```python
# 채택 정점 시점
print("\n" + "="*60)
print("모형의 주요 특성값")
print("="*60)

if estimated_q > estimated_p:
    t_peak = (1/(estimated_p + estimated_q)) * np.log(estimated_q/estimated_p)
    N_peak = bass_model(t_peak, estimated_m, estimated_p, estimated_q)
    n_peak = bass_density(t_peak, estimated_m, estimated_p, estimated_q)
    
    print(f"채택 정점 시점: {t_peak:.2f} 기")
    print(f"정점에서의 누적 채택자: {N_peak:.0f}")
    print(f"정점에서의 신규 채택자: {n_peak:.0f}")
    print(f"정점에서의 시장 침투율: {N_peak/estimated_m * 100:.1f}%")
    
    # 이론값과 비교
    theoretical_peak_rate = estimated_m * (estimated_p + estimated_q)**2 / (4 * estimated_q)
    print(f"이론적 정점 채택률: {theoretical_peak_rate:.0f} (검증)")
else:
    print("정점 없음 (p >= q인 경우)")

# 시장 포화 시점 계산 (95% 도달)
target_penetration = 0.95
t_saturation = -np.log((1 - target_penetration) * (1 + estimated_q/estimated_p)) / (estimated_p + estimated_q)
print(f"\n시장 포화 시점 (95% 도달): {t_saturation:.1f} 분기")

# 초기 채택자 중 혁신자 비율
innovator_ratio = estimated_p / (estimated_p + estimated_q)
print(f"초기 채택자 중 혁신자 비율: {innovator_ratio:.1%}")

# t=0에서의 초기 채택률
initial_rate = bass_density(0, estimated_m, estimated_p, estimated_q)
print(f"t=0에서의 초기 채택률: {initial_rate:.0f}")
```

### 6) 파라미터 민감도 분석

```python
print("\n" + "="*60)
print("파라미터 민감도 분석")
print("="*60)

# p 변화의 영향
p_variations = [estimated_p * 0.5, estimated_p, estimated_p * 1.5]
print("\n혁신계수(p) 변화의 영향:")
print("-" * 40)
for p_var in p_variations:
    adoption_20 = bass_model(20, estimated_m, p_var, estimated_q)
    ratio = p_var / estimated_p
    print(f"  p = {p_var:.4f} ({ratio:.1f}x): 20기 누적 = {adoption_20:,.0f} ({adoption_20/estimated_m*100:.1f}%)")

# q 변화의 영향  
q_variations = [estimated_q * 0.5, estimated_q, estimated_q * 1.5]
print("\n모방계수(q) 변화의 영향:")
print("-" * 40)
for q_var in q_variations:
    adoption_20 = bass_model(20, estimated_m, estimated_p, q_var)
    ratio = q_var / estimated_q
    print(f"  q = {q_var:.4f} ({ratio:.1f}x): 20기 누적 = {adoption_20:,.0f} ({adoption_20/estimated_m*100:.1f}%)")

print("\n" + "="*60)
print("분석 완료")
print("="*60)
```

## 6. BASS 모형의 확장

### 1) 일반화 BASS 모형 (Generalized Bass Model, GBM)

Bass, Krishnan & Jain (1994)이 제안한 확장 모형으로 마케팅 변수의 영향을 포함:

$$\frac{dN(t)}{dt} = [p + q \cdot F(t)] \cdot [m - N(t)] \cdot x(t)$$

여기서,

- x(t): 마케팅 믹스 변수 (가격, 광고 등)
- 일반적으로 x(t) = 1 + α₁P(t) + α₂A(t) 형태로 모델링
- P(t): 가격 변수, A(t): 광고 변수

### 2) 시변 파라미터 BASS 모형

파라미터가 시간에 따라 변하는 경우:

$$\frac{dN(t)}{dt} = p(t)(m - N(t)) + \frac{q(t)}{m} \cdot N(t) \cdot (m - N(t))$$

적용 사례:

- 제품 수명주기 단계별 다른 확산 패턴
- 계절성이 있는 제품
- 기술 발전에 따른 채택 행동 변화

### 3) 다중 세대 BASS 모형 (Multi-generation Model)

Norton & Bass (1987)의 세대 간 확산 모형:

$$\frac{dN_i(t)}{dt} = [p_i + q_i \cdot F_i(t)] \cdot [m_i(t) - N_i(t)]$$

여기서,

- i: 제품 세대 인덱스
- m_i(t): i세대의 시변 잠재시장 (이전 세대로부터 전환 포함)

### 4) 경쟁 BASS 모형 (Competitive Bass Model)

두 경쟁 제품의 동시 확산:

$$\frac{dN_1(t)}{dt} = (p_1 + q_{11}F_1(t) + q_{12}F_2(t))(m_1 - N_1(t))$$

$$\frac{dN_2(t)}{dt} = (p_2 + q_{21}F_1(t) + q_{22}F_2(t))(m_2 - N_2(t))$$

여기서,

- q_{ij}: 제품 j가 제품 i의 채택에 미치는 영향

### 5) 네트워크 BASS 모형

소셜 네트워크 구조를 반영:

$$\frac{dN_i(t)}{dt} = p_i(m_i - N_i(t)) + \sum_j w_{ij} \cdot q_i \cdot \frac{N_j(t)}{m_j} \cdot (m_i - N_i(t))$$

여기서,

- w_{ij}: 노드 j에서 노드 i로의 네트워크 가중치
- 지역별, 집단별 확산 패턴 모델링 가능

## 7. 실무 적용 가이드

### 1) 데이터 요구사항

**최소 데이터 요건**

- 시계열 길이: 최소 3개 시점, 권장 10개 이상
- 데이터 유형: 판매량, 가입자 수, 다운로드 수 등
- 데이터 주기: 일간, 주간, 월간, 분기별

**데이터 전처리**

- 이상치 제거: 3σ 규칙 또는 IQR 방법
- 결측치 보간: 선형 보간 또는 스플라인 보간
- 계절성 조정: X-12-ARIMA 또는 STL 분해

### 2) 파라미터 해석 가이드

**p/q 비율에 따른 제품 특성**

1. p/q < 0.1
    
    - 강한 구전효과 제품
    - 예: SNS 플랫폼, 메신저 앱
2. 0.1 ≤ p/q ≤ 0.5
    
    - 균형잡힌 확산
    - 예: 스마트폰, 태블릿
3. p/q > 0.5
    
    - 외부 마케팅 의존 제품
    - 예: 패션 제품, FMCG

**확산 속도 지표**

- (p+q) 값이 클수록 빠른 확산
- 일반적 범위: 0.3 ~ 0.6
- 하이테크 제품: 0.4 ~ 0.8
- 내구재: 0.2 ~ 0.4

### 3) 예측 정확도 향상 방법

**앙상블 접근법**

1. 복수 초기값으로 추정
2. 부트스트랩으로 신뢰구간 구성
3. 베이지안 추정으로 불확실성 정량화

**모형 검증**

- Hold-out 검증: 마지막 20% 데이터로 검증
- Rolling window: 시간 창을 이동하며 검증
- Cross-validation: k-fold 교차 검증

## 8. 주의사항 및 한계

### 1) BASS 모형의 한계

- 재구매 고려 안 함
- 경쟁 제품 영향 무시 (단일 제품 모형)
- 마케팅 믹스 변수 미반영 (기본 모형)
- 시장 규모 m의 사전 설정 어려움
- 제품 포기/이탈 미고려

### 2) 추정 시 주의사항

- 초기값 민감성: 여러 초기값 시도 필요
- 국소 최적해 문제: 전역 최적화 알고리즘 사용
- 데이터 부족 시 식별성 문제
- 과적합 위험: 파라미터 수 제한 필요

### 3) 실무 적용 팁

- 유사 제품의 과거 파라미터 참조
- 전문가 판단과 정량 모형 결합
- 정기적 재추정 및 모형 업데이트
- 시나리오 분석으로 불확실성 대응
- 단순 모형부터 시작, 필요시 확장

## 9. 참고문헌

### 핵심 논문 (오픈 엑세스)

Bass, F. M. (1969). A new product growth for model consumer durables. _Management Science_, _15_(5), 215-227. https://doi.org/10.1287/mnsc.15.5.215

Bass, F. M., Krishnan, T. V., & Jain, D. C. (1994). Why the Bass model fits without decision variables. _Marketing Science_, _13_(3), 203-223. https://doi.org/10.1287/mksc.13.3.203

Mahajan, V., Muller, E., & Bass, F. M. (1990). New product diffusion models in marketing: A review and directions for research. _Journal of Marketing_, _54_(1), 1-26. https://doi.org/10.1177/002224299005400101

Meade, N., & Islam, T. (2006). Modelling and forecasting the diffusion of innovation–A 25-year review. _International Journal of Forecasting_, _22_(3), 519-545. https://doi.org/10.1016/j.ijforecast.2006.01.005

Norton, J. A., & Bass, F. M. (1987). A diffusion theory model of adoption and substitution for successive generations of high-technology products. _Management Science_, _33_(9), 1069-1086. https://doi.org/10.1287/mnsc.33.9.1069

Peres, R., Muller, E., & Mahajan, V. (2010). Innovation diffusion and new product growth models: A critical review and research directions. _International Journal of Research in Marketing_, _27_(2), 91-106. https://doi.org/10.1016/j.ijresmar.2009.12.012

Van den Bulte, C., & Joshi, Y. V. (2007). New product diffusion with influentials and imitators. _Marketing Science_, _26_(3), 400-421. https://doi.org/10.1287/mksc.1060.0224

### 추가 학습 자료

Chandrasekaran, D., & Tellis, G. J. (2007). A critical review of marketing research on diffusion of new products. In N. K. Malhotra (Ed.), _Review of Marketing Research_ (Vol. 3, pp. 39-80). M.E. Sharpe. https://doi.org/10.1108/S1548-6435(2007)0000003006

Rogers, E. M. (2003). _Diffusion of innovations_ (5th ed.). Free Press.

Srinivasan, V., & Mason, C. H. (1986). Nonlinear least squares estimation of new product diffusion models. _Marketing Science_, _5_(2), 169-178. https://doi.org/10.1287/mksc.5.2.169