
## 목차

- [[#1. 시계열 분석의 기초 개념]]
- [[#2. ARIMA 모델의 구성 요소]]
- [[#3. ARIMA(p,d,q) 모델]]
- [[#4. SARIMA - 계절성 ARIMA]]
- [[#5. Python 실습 가이드]]
- [[#6. 실무 적용 사례]]
- [[#7. 고급 주제 및 확장]]
- [[#8. 학습 전략 및 팁]]
- [[#9. 학습 로드맵 및 다음 단계]]

---

## 1. 시계열 분석의 기초 개념

### 1.1 시계열 분석과 생성형 AI의 연결

시계열 분석은 시간에 따라 순차적으로 관측되는 데이터를 분석하는 방법입니다. 놀랍게도 우리가 사용하는 **생성형 AI도 본질적으로 시계열 모델**입니다.

💡 **생성형 AI 예시**: ChatGPT가 문장을 생성할 때, 이전 단어들을 바탕으로 다음에 올 가장 적절한 단어를 예측하는 과정은 시계열 예측과 동일한 원리입니다.

### 1.2 기본 선형회귀 vs 자기회귀

**일반 선형회귀**: $$Y_t = a + bX_t + \varepsilon_t$$

**자기회귀 (Auto Regressive)**: $$Y_t = a + bY_{t-1} + \varepsilon_t$$

핵심 차이점: 자기회귀는 **자기 자신의 과거 값**이 설명변수가 됩니다.

---

## 2. ARIMA 모델의 구성 요소

### 2.1 AR (Auto Regressive) - 자기회귀

#### 2.1.1 AR(1) 모델

$$Y_t = \mu + \phi(Y_{t-1} - \mu) + \varepsilon_t$$

또는 상수항 형태로: $$Y_t = c + \phi Y_{t-1} + \varepsilon_t$$

여기서:

- $\mu$: 장기 평균 (long-term mean)
- $\phi$: 자기회귀 계수 ($-1 < \phi < 1$이어야 안정)
- $c = \mu(1-\phi)$: 상수항
- $\varepsilon_t \sim N(0, \sigma^2)$: 백색잡음

💡 **주식 예시**: 오늘 주가 = 장기 평균 + 0.9 × (어제 주가 - 장기 평균) + 오늘의 충격

#### 2.1.2 장기 평균의 의미

$$\mu = \frac{c}{1 - \phi}$$

💡 **해석 예시**: $\phi = 0.9$, $c = 1$인 경우

- 장기 평균 = $\frac{1}{1-0.9} = 10$
- 현재 값이 15라면, 시간이 지나면서 점진적으로 10으로 수렴

#### 2.1.3 동적 예측 (Dynamic Forecasting)

$$\begin{align} F_{t+1} &= c + \phi Y_t \ F_{t+2} &= c + \phi F_{t+1} \ F_{t+3} &= c + \phi F_{t+2} \end{align}$$

💡 **핵심**: 예측값을 다시 입력으로 사용하여 미래를 계속 예측할 수 있습니다.

#### 2.1.4 AR(p) 모델로 확장

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \varepsilon_t$$

#### 2.1.5 반감기 (Half-life)

AR(1) 모델에서 충격이 절반으로 줄어드는 데 걸리는 시간: $$\text{반감기} = \frac{\ln(0.5)}{\ln(\phi)}$$

💡 **예시**: $\phi = 0.9$인 경우 반감기 ≈ 6.6기간

### 2.2 백시프트 연산자 (Backshift Operator)

시계열 모델을 간결하게 표현하는 수학적 도구입니다.

#### 2.2.1 백시프트 연산자 정의

$$B \cdot Y_t = Y_{t-1}$$ $$B^2 \cdot Y_t = Y_{t-2}$$ $$B^k \cdot Y_t = Y_{t-k}$$

#### 2.2.2 AR(1) 모델의 백시프트 표현

AR(1) 모델: $$Y_t = c + \phi Y_{t-1} + \varepsilon_t$$

이를 정리하면: $$Y_t - \phi Y_{t-1} = c + \varepsilon_t$$

백시프트 연산자를 사용하여: $$Y_t - \phi B Y_t = c + \varepsilon_t$$ $$(1 - \phi B)Y_t = c + \varepsilon_t$$

#### 2.2.3 복잡한 모델의 간단한 표현

계절성 + 일반 AR 모델: $$(1 - \phi B)(1 - \Phi B^{12})Y_t = c + \varepsilon_t$$

전개하면: $$Y_t - \phi Y_{t-1} - \Phi Y_{t-12} + \phi\Phi Y_{t-13} = c + \varepsilon_t$$

💡 **해석**: 전월 값($Y_{t-1}$)과 전년 동월 값($Y_{t-12}$)이 모두 현재 값에 영향을 줌

### 2.3 MA (Moving Average) - 이동평균

#### 2.3.1 MA(1) 모델

$$Y_t = \mu + \varepsilon_t - \theta \varepsilon_{t-1}$$

💡 **오해 주의**: 일반적인 이동평균(값들의 평균)과는 다름. 오차항들의 선형결합임.

#### 2.3.2 백시프트 연산자로 표현

$$Y_t = \mu + (1 - \theta B)\varepsilon_t$$

#### 2.3.3 MA(q) 모델

$$Y_t = \mu + \varepsilon_t - \theta_1\varepsilon_{t-1} - \theta_2\varepsilon_{t-2} - \cdots - \theta_q\varepsilon_{t-q}$$

백시프트 연산자로: $$Y_t = \mu + (1 - \theta_1 B - \theta_2 B^2 - \cdots - \theta_q B^q)\varepsilon_t$$

#### 2.3.4 AR과 MA의 관계

놀랍게도:

- **AR(1) = MA(∞)**: AR(1) 모델은 무한한 MA 모델과 동일
- **MA(1) = AR(∞)**: MA(1) 모델은 무한한 AR 모델과 동일

AR(1) 모델 $(1 - \phi B)Y_t = \varepsilon_t$를 전개하면: $$Y_t = \frac{1}{1-\phi B}\varepsilon_t = (1 + \phi B + \phi^2 B^2 + \cdots)\varepsilon_t$$ $$Y_t = \varepsilon_t + \phi \varepsilon_{t-1} + \phi^2 \varepsilon_{t-2} + \phi^3 \varepsilon_{t-3} + \cdots$$

### 2.4 I (Integrated) - 차분과 단위근

#### 2.4.1 단위근 문제

$\phi = 1$인 경우 (단위근 존재): $$Y_t = c + Y_{t-1} + \varepsilon_t$$

이는 **랜덤워크**가 되어:

- 장기 평균이 존재하지 않음
- 과거 충격이 영구히 지속
- 분산이 시간에 따라 증가: $\text{Var}(Y_t) = t \sigma^2$
- 예측이 매우 어려움

💡 **주가 예시**: 대부분의 주가는 랜덤워크 특성을 보임. 오늘 주가가 내일 주가의 최선의 예측치.

#### 2.4.2 차분 (Differencing)

해결책: 차분을 통해 안정적 시계열로 변환 $$\begin{align} \text{1차 차분: } \Delta Y_t &= Y_t - Y_{t-1} = (1-B)Y_t \ \text{2차 차분: } \Delta^2 Y_t &= \Delta Y_t - \Delta Y_{t-1} = (1-B)^2Y_t \end{align}$$

💡 **주식 수익률**: 로그 주가의 차분 = 수익률 (안정적 특성) $$r_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right) \approx \frac{P_t - P_{t-1}}{P_{t-1}}$$

#### 2.4.3 단위근 검정

**ADF (Augmented Dickey-Fuller) 검정**: $$\Delta Y_t = \alpha + \gamma Y_{t-1} + \sum_{i=1}^p \beta_i \Delta Y_{t-i} + \varepsilon_t$$

- $H_0$: $\gamma = 0$ (단위근 존재)
- $H_1$: $\gamma < 0$ (안정적 시계열)

---

## 3. ARIMA(p,d,q) 모델

### 3.1 모델 표기법

**ARIMA(p,d,q)**:

- **p**: AR 차수 (과거 몇 개 관측값 사용)
- **d**: 차분 차수 (몇 번 차분)
- **q**: MA 차수 (과거 몇 개 오차항 사용)

### 3.2 일반형 ARIMA 모델

$$\phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t$$

여기서: $$\begin{align} \phi(B) &= 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p \ \theta(B) &= 1 - \theta_1 B - \theta_2 B^2 - \cdots - \theta_q B^q \end{align}$$

### 3.3 모델 선택 기준

#### 3.3.1 정보 기준 (Information Criteria)

**AIC (Akaike Information Criterion)**: $$\text{AIC} = -2\ln(L) + 2k$$

**BIC (Bayesian Information Criterion)**: $$\text{BIC} = -2\ln(L) + k \ln(n)$$

**AICc (Corrected AIC)**: $$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

여기서:

- $L$: 우도함수
- $k$: 모수 개수
- $n$: 표본 크기

💡 **모델 선택**: AIC, BIC가 작을수록 좋은 모델. 과적합을 방지하기 위해 모수 개수에 페널티 부여.

#### 3.3.2 Box-Jenkins 방법론

1. **식별 (Identification)**: ACF, PACF 분석으로 p, d, q 결정
2. **추정 (Estimation)**: 최대우도법으로 모수 추정
3. **진단 (Diagnostic)**: 잔차 분석으로 모델 적합성 검토

---

## 4. SARIMA - 계절성 ARIMA

### 4.1 SARIMA(p,d,q)(P,D,Q)ₛ 모델

#### 4.1.1 표기법 의미

- **(p,d,q)**: 일반적인 ARIMA 성분
- **(P,D,Q)ₛ**: 계절적 ARIMA 성분
- **s**: 계절 주기 (월별=12, 분기별=4, 주별=52)

#### 4.1.2 계절성 AR 예시

$$Y_t = c + \phi Y_{t-1} + \Phi Y_{t-12} + \varepsilon_t$$

💡 **해석**: 이번 달 매출 = 상수 + (지난 달 매출 영향) + (작년 같은 달 매출 영향) + 충격

#### 4.1.3 일반형 SARIMA 모델

$$\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D Y_t = \theta(B)\Theta(B^s)\varepsilon_t$$

여기서: $$\begin{align} \Phi(B^s) &= 1 - \Phi_1 B^s - \Phi_2 B^{2s} - \cdots - \Phi_P B^{Ps} \ \Theta(B^s) &= 1 - \Theta_1 B^s - \Theta_2 B^{2s} - \cdots - \Theta_Q B^{Qs} \end{align}$$

#### 4.1.4 SARIMA(1,1,1)(1,1,1)₁₂ 예시

$$(1-\phi B)(1-\Phi B^{12})(1-B)(1-B^{12})Y_t = (1-\theta B)(1-\Theta B^{12})\varepsilon_t$$

전개하면 매우 복잡한 식이 되지만, 백시프트 연산자를 사용하면 간결하게 표현 가능합니다.

### 4.2 계절성 차분

**일반 차분**: $$\nabla Y_t = (1-B)Y_t = Y_t - Y_{t-1}$$

**계절 차분**: $$\nabla_s Y_t = (1-B^s)Y_t = Y_t - Y_{t-s}$$

💡 **예시**: 월별 데이터에서 $\nabla_{12} Y_t = Y_t - Y_{t-12}$ (전년 동월 대비 변화)

**이중 차분** (일반 + 계절): $$\nabla \nabla_s Y_t = (1-B)(1-B^s)Y_t$$

### 4.3 계절성 분해

시계열 $Y_t$를 다음과 같이 분해: $$Y_t = T_t + S_t + R_t$$

여기서:

- $T_t$: 추세 성분 (Trend)
- $S_t$: 계절성 성분 (Seasonal)
- $R_t$: 불규칙 성분 (Remainder)

**승법 모델**: $$Y_t = T_t \times S_t \times R_t$$

---

## 5. Python 실습 가이드


---

## 마무리

### 핵심 요약

ARIMA와 SARIMA는 시계열 분석의 핵심 도구로, 다음과 같은 특징을 가집니다:

**ARIMA의 핵심 구성요소**:

- **AR (Auto Regressive)**: 과거 값으로 현재 값 예측
- **I (Integrated)**: 차분을 통한 정상성 확보
- **MA (Moving Average)**: 과거 오차로 현재 값 설명

**SARIMA의 확장**:

- 계절성 패턴을 추가로 고려
- 복잡한 시계열 패턴 모델링 가능
- 실무에서 높은 활용도

### 학습 전략

1. **시뮬레이션 우선**: 답을 알고 있는 데이터로 먼저 검증
2. **단계적 접근**: 간단한 모델부터 복잡한 모델로
3. **실무 감각**: 통계적 유의성과 비즈니스 의미를 함께 고려
4. **지속적 학습**: 새로운 방법론과의 결합 및 확장

### 다음 단계

- **VECM (Vector Error Correction Model)**: 다변량 시계열 분석
- **딥러닝 시계열**: LSTM, Transformer 등 최신 기법
- **실시간 시스템**: 프로덕션 환경에서의 시계열 분석
- **도메인 특화**: 본인 전공 분야에 특화된 시계열 분석

💡 **마지막 조언**: 이론을 이해하고 충분한 실습을 통해 실무에서 활용할 수 있는 역량을 기르시기 바랍니다. 특히 **왜 이 모델을 선택했는지**, **결과를 어떻게 해석할지**에 대한 깊이 있는 사고가 중요합니다.

---

**과제**: 제공된 코드 템플릿을 사용하여 본인 관심 분야의 시계열 데이터를 분석하고, ARIMA/SARIMA 모델을 구축해보세요. 모델 선택 이유와 결과 해석을 포함한 보고서를 작성해보시기 바랍니다.

---

> **참고사항**: 이 강의노트는 옵시디언에서 사용하기 위해 작성되었습니다. `[[링크]]` 문법을 사용하여 섹션 간 이동이 가능하며, 태그(`#시계열분석`, `#ARIMA` 등)를 통해 관련 내용을 쉽게 찾을 수 있습니다.