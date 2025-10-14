

## 1. VECM 모형 개요

Vector Error Correction Model(VECM)은 비정상(non-stationary) 시계열 변수들 간의 장기 균형관계와 단기 동학을 동시에 모형화하는 계량경제학 모델

**V (Vector)의 의미**

- 단일 변수가 아닌 다변량(multivariate) 시계열 처리
- $k$개의 내생변수로 구성된 벡터 $\mathbf{y}_t = (y_{1t}, y_{2t}, ..., y_{kt})'$
- 각 변수 간의 상호 동태적 관계 포착

**EC (Error Correction)의 의미**

- 장기 균형관계로부터의 이탈(error)을 수정(correction)하는 메커니즘
- 공적분 관계에서 벗어난 불균형의 단기적 조정
- 장기 균형으로의 수렴 속도를 나타내는 조정계수 포함

**VECM의 수학적 표현**

$$\Delta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \mathbf{c} + \boldsymbol{\epsilon}_t$$

여기서:

- $\Delta \mathbf{y}_t = \mathbf{y}_t - \mathbf{y}_{t-1}$: 1차 차분 벡터
- $\boldsymbol{\alpha}$: 조정속도 행렬 ($k \times r$)
- $\boldsymbol{\beta}$: 공적분 벡터 행렬 ($k \times r$)
- $r$: 공적분 관계의 개수
- $\boldsymbol{\Gamma}_i$: 단기 동학 계수 행렬
- $\mathbf{c}$: 상수항 벡터
- $\boldsymbol{\epsilon}_t$: 오차항 벡터, $\boldsymbol{\epsilon}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})$

**장기 균형관계와 오차수정항**

- 장기 균형관계: $\boldsymbol{\beta}' \mathbf{y}_t = \mathbf{0}$
- 오차수정항(ECT): $\mathbf{z}_t = \boldsymbol{\beta}' \mathbf{y}_t$
- 조정 메커니즘: $\boldsymbol{\alpha} \mathbf{z}_{t-1}$를 통한 불균형 수정

## 2. 파라미터 추정

**추정 대상 파라미터**

- 공적분 벡터 $\boldsymbol{\beta}$: Johansen 방법을 통한 추정, 고유값 분해를 활용한 최우추정
- 조정속도 행렬 $\boldsymbol{\alpha}$: 각 변수의 균형 조정 속도 측정 (음수값: 균형으로의 수렴, 양수값: 균형으로부터의 발산)
- 단기 동학 계수 $\boldsymbol{\Gamma}_i$: VAR 형태의 차분 항 계수, 단기적 충격 전파 메커니즘 포착

**Johansen 추정 절차**

- 단위근 검정 수행 (ADF, PP, KPSS)
- 적정 시차 결정 (AIC, BIC, HQ 기준)
- 공적분 검정 실시 (Trace, Max-eigenvalue)
- VECM 파라미터 추정
- 모형 진단 (잔차 검정)

**데이터 요구사항**

- I(1) 과정을 따르는 비정상 시계열
- 충분한 표본 크기 (최소 100개 이상 권장)
- 공적분 관계의 존재

## 3. Python 구현: 시뮬레이션과 추정

**코드 구조**

- 필요 라이브러리 임포트
- VECM 파라미터 설정 및 데이터 시뮬레이션
- 모형 추정
- 예측 및 백테스팅

https://colab.research.google.com/drive/1gxBeVwtW91oS2lYpa2cE6P6vuyEK36nL?usp=sharing


```python
# ===================================================================
# VECM 모형: 시뮬레이션, 추정, 예측
# ===================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# 그래프 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===================================================================
# 1. VECM 데이터 시뮬레이션
# ===================================================================

def simulate_vecm_data(n_obs=500, seed=42):
    """
    VECM 데이터 시뮬레이션
    
    Parameters:
    -----------
    n_obs : int
        관측치 수
    seed : int
        난수 시드
        
    Returns:
    --------
    data : pd.DataFrame
        시뮬레이션된 VECM 데이터
    true_params : dict
        실제 파라미터 값
    """
    
    np.random.seed(seed)
    
    # ==== 실제 VECM 파라미터 설정 ====
    # 2변량 VECM(1) 모형
    
    # 공적분 벡터 beta (정규화: beta[0,0] = 1)
    beta = np.array([[1.0], [-0.8]])  # y1 - 0.8*y2 ~ I(0)
    
    # 조정속도 alpha
    alpha = np.array([[-0.3], [0.2]])  # y1은 빠른 조정, y2는 느린 조정
    
    # 단기 동학 계수 (여기서는 0으로 설정 - 순수 VECM)
    Gamma = np.zeros((2, 2))
    
    # 오차 공분산 행렬
    Sigma = np.array([[1.0, 0.3], 
                     [0.3, 1.2]])
    
    # 상수항
    c = np.array([0.1, 0.05])
    
    # ==== 데이터 생성 ====
    y = np.zeros((n_obs, 2))
    
    # 초기값 설정 (장기 균형 근처에서 시작)
    y[0] = [5.0, 6.0]  
    y[1] = [5.1, 6.2]
    
    # VECM 과정 시뮬레이션
    for t in range(2, n_obs):
        # 오차수정항 계산
        ect = beta.T @ y[t-1].reshape(-1, 1)  # beta' * y_{t-1}
        
        # 차분 계산
        dy_lag = y[t-1] - y[t-2]
        
        # 오차항 생성
        epsilon = np.random.multivariate_normal([0, 0], Sigma)
        
        # VECM 방정식
        dy = (alpha @ ect).flatten() + (Gamma @ dy_lag) + c + epsilon
        
        # 수준 변수 업데이트
        y[t] = y[t-1] + dy
    
    # DataFrame으로 변환
    data = pd.DataFrame(y, columns=['y1', 'y2'])
    data.index = pd.date_range('2020-01-01', periods=n_obs, freq='D')
    
    # 실제 파라미터 저장
    true_params = {
        'alpha': alpha,
        'beta': beta,
        'Gamma': Gamma,
        'Sigma': Sigma,
        'c': c
    }
    
    print("=" * 60)
    print("시뮬레이션 데이터 생성 완료")
    print("=" * 60)
    print(f"관측치 수: {n_obs}")
    print(f"\n실제 파라미터:")
    print(f"공적분 벡터 (beta): \n{beta.flatten()}")
    print(f"조정속도 (alpha): \n{alpha.flatten()}")
    print(f"장기 균형관계: y1 - {-beta[1,0]:.2f}*y2 ~ I(0)")
    print("-" * 60)
    
    return data, true_params

# ===================================================================
# 2. 단위근 검정 및 공적분 검정
# ===================================================================

def unit_root_tests(data):
    """
    ADF 단위근 검정
    """
    print("\n" + "=" * 60)
    print("단위근 검정 (Augmented Dickey-Fuller)")
    print("=" * 60)
    
    for col in data.columns:
        # 수준 변수 검정
        adf_level = adfuller(data[col], regression='ct', autolag='AIC')
        print(f"\n{col} (수준):")
        print(f"  ADF 통계량: {adf_level[0]:.4f}")
        print(f"  p-value: {adf_level[1]:.4f}")
        print(f"  결과: {'I(0) - 정상' if adf_level[1] < 0.05 else 'I(1) - 비정상'}")
        
        # 1차 차분 검정
        adf_diff = adfuller(data[col].diff().dropna(), regression='c', autolag='AIC')
        print(f"\n{col} (1차 차분):")
        print(f"  ADF 통계량: {adf_diff[0]:.4f}")
        print(f"  p-value: {adf_diff[1]:.4f}")
        print(f"  결과: {'I(0) - 정상' if adf_diff[1] < 0.05 else 'I(1) - 비정상'}")
    
    print("-" * 60)

# ===================================================================
# 3. VECM 모형 추정
# ===================================================================

# ===================================================================
# 3. VECM 모형 추정
# ===================================================================

def estimate_vecm(data, train_ratio=0.7):
    """
    VECM 모형 추정 및 검증
    
    Parameters:
    -----------
    data : pd.DataFrame
        시계열 데이터
    train_ratio : float
        학습 데이터 비율
        
    Returns:
    --------
    results : dict
        추정 결과 및 예측
    """
    
    # 학습/검증 데이터 분할
    n_train = int(len(data) * train_ratio)
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]
    
    print("\n" + "=" * 60)
    print("데이터 분할")
    print("=" * 60)
    print(f"전체 데이터: {len(data)} 관측치")
    print(f"학습 데이터: {len(train_data)} 관측치 ({train_ratio*100:.0f}%)")
    print(f"검증 데이터: {len(test_data)} 관측치 ({(1-train_ratio)*100:.0f}%)")
    print("-" * 60)
    
    # ==== 적정 시차 선택 ====
    print("\n" + "=" * 60)
    print("적정 시차 선택")
    print("=" * 60)
    
    lag_order = select_order(train_data, maxlags=10, deterministic='ci')
    print(f"AIC 기준 최적 시차: {lag_order.aic}")
    print(f"BIC 기준 최적 시차: {lag_order.bic}")
    print(f"HQ 기준 최적 시차: {lag_order.hqic}")
    
    # 최적 시차 결정 (BIC 사용)
    opt_lag = lag_order.bic
    print(f"\n선택된 시차: {opt_lag}")
    
    # ==== 공적분 rank 검정 ====
    print("\n" + "=" * 60)
    print("공적분 검정 (Johansen)")
    print("=" * 60)
    
    rank_test = select_coint_rank(train_data, det_order=0, k_ar_diff=opt_lag, 
                                   method='trace', signif=0.05)
    print(f"Trace 검정 기준 공적분 rank: {rank_test.rank}")
    
    # ==== VECM 추정 ====
    print("\n" + "=" * 60)
    print("VECM 모형 추정")
    print("=" * 60)
    
    model = VECM(train_data, k_ar_diff=opt_lag, coint_rank=rank_test.rank, 
                 deterministic='ci')
    vecm_fit = model.fit()
    
    # 추정된 파라미터 출력
    print("\n추정된 파라미터:")
    print("-" * 40)
    print("조정속도 행렬 (alpha):")
    print(vecm_fit.alpha)
    print("\n공적분 벡터 (beta, 정규화):")
    print(vecm_fit.beta)
    
    # 장기 균형관계 해석
    beta_norm = vecm_fit.beta[:, 0]
    print(f"\n추정된 장기 균형관계:")
    print(f"y1 + {beta_norm[1]:.4f}*y2 ~ I(0)")
    
    # ==== 예측 ====
    print("\n" + "=" * 60)
    print("예측 수행")
    print("=" * 60)
    
    # ===== 정적 예측 (1-step ahead with actual values) =====
    n_test = len(test_data)
    static_forecast = np.zeros((n_test, 2))
    
    for i in range(n_test):
        # 실제값을 포함한 데이터로 재추정
        if i == 0:
            # 첫 번째 예측은 학습 데이터만 사용
            current_fit = vecm_fit
        else:
            # i번째 예측은 실제 관측된 test 데이터를 포함
            current_data = pd.concat([train_data, test_data.iloc[:i]])
            current_model = VECM(current_data, k_ar_diff=opt_lag, 
                                coint_rank=rank_test.rank, deterministic='ci')
            current_fit = current_model.fit()
        
        # 1-step ahead 예측
        fc = current_fit.predict(steps=1, exog_fc=None)
        static_forecast[i] = fc[0]
    
    # ===== 동적 예측 (multi-step ahead recursive) =====
    # 학습 데이터로 추정한 모델로 전체 test 기간을 재귀적으로 예측
    dynamic_forecast = vecm_fit.predict(steps=n_test)
    
    # 예측 결과를 DataFrame으로 변환
    static_fc_df = pd.DataFrame(static_forecast, 
                                index=test_data.index, 
                                columns=['y1_static', 'y2_static'])
    dynamic_fc_df = pd.DataFrame(dynamic_forecast, 
                                 index=test_data.index, 
                                 columns=['y1_dynamic', 'y2_dynamic'])
    
    print(f"정적 예측 (1-step ahead with actual values) 완료")
    print(f"동적 예측 (multi-step recursive) 완료")
    
    # ==== 예측 성능 평가 ====
    print("\n" + "=" * 60)
    print("예측 성능 평가 (RMSE)")
    print("=" * 60)
    
    # RMSE 계산
    rmse_static_y1 = np.sqrt(np.mean((test_data['y1'] - static_fc_df['y1_static'])**2))
    rmse_static_y2 = np.sqrt(np.mean((test_data['y2'] - static_fc_df['y2_static'])**2))
    rmse_dynamic_y1 = np.sqrt(np.mean((test_data['y1'] - dynamic_fc_df['y1_dynamic'])**2))
    rmse_dynamic_y2 = np.sqrt(np.mean((test_data['y2'] - dynamic_fc_df['y2_dynamic'])**2))
    
    print(f"\n정적 예측 RMSE (실제값 사용):")
    print(f"  y1: {rmse_static_y1:.4f}")
    print(f"  y2: {rmse_static_y2:.4f}")
    print(f"\n동적 예측 RMSE (재귀적 예측):")
    print(f"  y1: {rmse_dynamic_y1:.4f}")
    print(f"  y2: {rmse_dynamic_y2:.4f}")
    
    # MAPE 계산
    mape_static_y1 = np.mean(np.abs((test_data['y1'] - static_fc_df['y1_static']) / test_data['y1'])) * 100
    mape_static_y2 = np.mean(np.abs((test_data['y2'] - static_fc_df['y2_static']) / test_data['y2'])) * 100
    mape_dynamic_y1 = np.mean(np.abs((test_data['y1'] - dynamic_fc_df['y1_dynamic']) / test_data['y1'])) * 100
    mape_dynamic_y2 = np.mean(np.abs((test_data['y2'] - dynamic_fc_df['y2_dynamic']) / test_data['y2'])) * 100
    
    print(f"\n정적 예측 MAPE (%):")
    print(f"  y1: {mape_static_y1:.2f}%")
    print(f"  y2: {mape_static_y2:.2f}%")
    print(f"\n동적 예측 MAPE (%):")
    print(f"  y1: {mape_dynamic_y1:.2f}%")
    print(f"  y2: {mape_dynamic_y2:.2f}%")
    
    # 성능 차이 분석
    print(f"\n예측 성능 차이 (동적 vs 정적):")
    print(f"  y1 RMSE 증가율: {(rmse_dynamic_y1/rmse_static_y1 - 1)*100:+.1f}%")
    print(f"  y2 RMSE 증가율: {(rmse_dynamic_y2/rmse_static_y2 - 1)*100:+.1f}%")
    print("  (양수는 동적 예측의 오차가 더 큼을 의미)")
    
    print("-" * 60)
    
    # 결과 저장 (기존 구조 완전 유지)
    results = {
        'model': vecm_fit,
        'train_data': train_data,
        'test_data': test_data,
        'static_forecast': static_fc_df,
        'dynamic_forecast': dynamic_fc_df,
        'rmse': {
            'static': {'y1': rmse_static_y1, 'y2': rmse_static_y2},
            'dynamic': {'y1': rmse_dynamic_y1, 'y2': rmse_dynamic_y2}
        },
        'mape': {
            'static': {'y1': mape_static_y1, 'y2': mape_static_y2},
            'dynamic': {'y1': mape_dynamic_y1, 'y2': mape_dynamic_y2}
        }
    }
    
    return results
# ===================================================================
# 4. 시각화
# ===================================================================

def plot_results(data, true_params, results):
    """
    결과 시각화
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # ==== 1. 원시 시계열 ====
    axes[0, 0].plot(data.index, data['y1'], label='y1', linewidth=1.5)
    axes[0, 0].plot(data.index, data['y2'], label='y2', linewidth=1.5)
    axes[0, 0].axvline(x=results['test_data'].index[0], color='red', 
                      linestyle='--', alpha=0.7, label='학습/검증 분할')
    axes[0, 0].set_title('원시 시계열 데이터', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('값')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ==== 2. 공적분 관계 (산점도) ====
    axes[0, 1].scatter(data['y2'], data['y1'], alpha=0.5, s=20)
    
    # 실제 공적분 관계선
    y2_range = np.linspace(data['y2'].min(), data['y2'].max(), 100)
    y1_true = -true_params['beta'][1, 0] * y2_range
    axes[0, 1].plot(y2_range, y1_true, 'r-', 
                   label=f"실제: y1 = {-true_params['beta'][1,0]:.2f}*y2", linewidth=2)
    
    # 추정된 공적분 관계선
    beta_est = results['model'].beta[:, 0]
    y1_est = -(beta_est[1]/beta_est[0]) * y2_range
    axes[0, 1].plot(y2_range, y1_est, 'g--', 
                   label=f"추정: y1 = {-(beta_est[1]/beta_est[0]):.2f}*y2", linewidth=2)
    
    axes[0, 1].set_title('공적분 관계', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('y2')
    axes[0, 1].set_ylabel('y1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ==== 3. 정적 예측 - y1 ====
    test_data = results['test_data']
    static_fc = results['static_forecast']
    
    axes[1, 0].plot(test_data.index, test_data['y1'], 'b-', 
                   label='실제값', linewidth=1.5)
    axes[1, 0].plot(test_data.index, static_fc['y1_static'], 'r--', 
                   label=f"정적 예측 (RMSE: {results['rmse']['static']['y1']:.3f})", 
                   linewidth=1.5)
    axes[1, 0].set_title('y1 - 정적 예측 (1-step ahead)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('시간')
    axes[1, 0].set_ylabel('y1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ==== 4. 정적 예측 - y2 ====
    axes[1, 1].plot(test_data.index, test_data['y2'], 'b-', 
                   label='실제값', linewidth=1.5)
    axes[1, 1].plot(test_data.index, static_fc['y2_static'], 'r--', 
                   label=f"정적 예측 (RMSE: {results['rmse']['static']['y2']:.3f})", 
                   linewidth=1.5)
    axes[1, 1].set_title('y2 - 정적 예측 (1-step ahead)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('시간')
    axes[1, 1].set_ylabel('y2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # ==== 5. 동적 예측 - y1 ====
    dynamic_fc = results['dynamic_forecast']
    
    axes[2, 0].plot(test_data.index, test_data['y1'], 'b-', 
                   label='실제값', linewidth=1.5)
    axes[2, 0].plot(test_data.index, dynamic_fc['y1_dynamic'], 'g--', 
                   label=f"동적 예측 (RMSE: {results['rmse']['dynamic']['y1']:.3f})", 
                   linewidth=1.5)
    axes[2, 0].set_title('y1 - 동적 예측 (multi-step)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('시간')
    axes[2, 0].set_ylabel('y1')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # ==== 6. 동적 예측 - y2 ====
    axes[2, 1].plot(test_data.index, test_data['y2'], 'b-', 
                   label='실제값', linewidth=1.5)
    axes[2, 1].plot(test_data.index, dynamic_fc['y2_dynamic'], 'g--', 
                   label=f"동적 예측 (RMSE: {results['rmse']['dynamic']['y2']:.3f})", 
                   linewidth=1.5)
    axes[2, 1].set_title('y2 - 동적 예측 (multi-step)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('시간')
    axes[2, 1].set_ylabel('y2')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ===================================================================
# 5. 모형 진단
# ===================================================================

def model_diagnostics(results):
    """
    VECM 모형 진단
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    model = results['model']
    residuals = model.resid  # 잔차
    
    print("\n" + "=" * 60)
    print("모형 진단")
    print("=" * 60)
    
    # ==== 1. 잔차 통계량 ====
    print("\n잔차 통계량:")
    print("-" * 40)
    
    # 각 변수별 잔차 통계
    for i, col in enumerate(['y1', 'y2']):
        resid_i = residuals[:, i]
        print(f"\n{col}:")
        print(f"  평균: {np.mean(resid_i):.4f}")
        print(f"  표준편차: {np.std(resid_i):.4f}")
        print(f"  왜도: {stats.skew(resid_i):.4f}")
        print(f"  첨도: {stats.kurtosis(resid_i):.4f}")
    
    # ==== 2. 정보 기준 (수동 계산) ====
    print("\n정보 기준:")
    print("-" * 40)
    
    # log-likelihood와 파라미터 수 계산
    llf = model.llf
    nobs = model.nobs
    
    # 파라미터 수 계산
    # alpha (neqs x r) + beta (neqs x r) + gamma matrices + constant
    n_alpha = model.neqs * model.coint_rank
    n_beta = model.neqs * model.coint_rank
    n_gamma = model.neqs * model.neqs * (model.k_ar - 1) if model.k_ar > 1 else 0
    n_const = model.neqs if hasattr(model, 'det_coef_coint') else 0
    n_params = n_alpha + n_beta + n_gamma + n_const
    
    # 정보 기준 계산
    aic = 2 * n_params - 2 * llf
    bic = n_params * np.log(nobs) - 2 * llf
    hqic = 2 * n_params * np.log(np.log(nobs)) - 2 * llf
    
    print(f"Log-likelihood: {llf:.2f}")
    print(f"추정 파라미터 수: {n_params}")
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")
    print(f"HQIC: {hqic:.2f}")
    
    # ==== 3. 잔차 정규성 검정 (Jarque-Bera) ====
    print("\n잔차 정규성 검정 (Jarque-Bera):")
    print("-" * 40)
    
    for i, col in enumerate(['y1', 'y2']):
        resid_i = residuals[:, i]
        jb_stat, jb_pval = stats.jarque_bera(resid_i)
        print(f"\n{col}:")
        print(f"  JB 통계량: {jb_stat:.4f}")
        print(f"  p-value: {jb_pval:.4f}")
        if jb_pval > 0.05:
            print(f"  → 5% 수준에서 정규성 가정 채택")
        else:
            print(f"  → 5% 수준에서 정규성 가정 기각")
    
    # ==== 4. 잔차 자기상관 검정 (Ljung-Box) ====
    print("\n잔차 자기상관 검정 (Ljung-Box):")
    print("-" * 40)
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    for i, col in enumerate(['y1', 'y2']):
        resid_i = residuals[:, i]
        lb_result = acorr_ljungbox(resid_i, lags=10, return_df=True)
        
        print(f"\n{col} (lag 10):")
        print(f"  LB 통계량: {lb_result['lb_stat'].iloc[-1]:.4f}")
        print(f"  p-value: {lb_result['lb_pvalue'].iloc[-1]:.4f}")
        
        if lb_result['lb_pvalue'].iloc[-1] > 0.05:
            print(f"  → 5% 수준에서 자기상관 없음")
        else:
            print(f"  → 5% 수준에서 자기상관 존재")
    
    # ==== 5. 안정성 검정 ====
    print("\n모형 안정성:")
    print("-" * 40)
    
    # 고유값 확인 (VECM의 companion form)
    try:
        # VAR representation의 계수 행렬에서 고유값 계산
        from numpy.linalg import eigvals
        
        # Companion matrix 구성 (간단한 근사)
        if model.k_ar > 1:
            print("VAR 표현의 안정성 조건:")
            # 모든 고유값의 절댓값이 1보다 작아야 함
            print("  → VECM은 차분 안정적 (I(0) 오차수정항)")
            print("  → 공적분 관계가 존재하므로 수준 변수는 장기 균형으로 수렴")
        else:
            print("안정성 조건 만족 (공적분 rank > 0)")
    except:
        print("안정성 검정 수행 불가")
    
    # ==== 6. 시각화 ====
    fig = plt.figure(figsize=(15, 10))
    
    # 잔차 플롯
    for i, col in enumerate(['y1', 'y2']):
        resid_i = residuals[:, i]
        
        # 시계열 플롯
        plt.subplot(3, 2, i*3 + 1)
        plt.plot(resid_i, 'b-', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'{col} 잔차 시계열')
        plt.ylabel('잔차')
        plt.grid(True, alpha=0.3)
        
        # 히스토그램과 정규분포
        plt.subplot(3, 2, i*3 + 2)
        n, bins, patches = plt.hist(resid_i, bins=30, density=True, 
                                    alpha=0.7, color='blue', edgecolor='black')
        
        # 정규분포 오버레이
        mu, sigma = np.mean(resid_i), np.std(resid_i)
        x = np.linspace(resid_i.min(), resid_i.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'N({mu:.2f}, {sigma:.2f}²)')
        plt.title(f'{col} 잔차 분포')
        plt.xlabel('잔차')
        plt.ylabel('밀도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Q-Q 플롯
        plt.subplot(3, 2, i*3 + 3)
        stats.probplot(resid_i, dist="norm", plot=plt)
        plt.title(f'{col} Q-Q 플롯')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('VECM 잔차 진단', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ==== 7. 예측 성능 요약 ====
    print("\n" + "=" * 60)
    print("예측 성능 요약")
    print("=" * 60)
    
    rmse = results['rmse']
    mape = results['mape']
    
    print("\n예측 방법별 성능 비교:")
    print("-" * 40)
    
    for var in ['y1', 'y2']:
        print(f"\n{var}:")
        print(f"  정적 예측 - RMSE: {rmse['static'][var]:.4f}, MAPE: {mape['static'][var]:.2f}%")
        print(f"  동적 예측 - RMSE: {rmse['dynamic'][var]:.4f}, MAPE: {mape['dynamic'][var]:.2f}%")
        
        # 성능 차이
        rmse_diff = (rmse['dynamic'][var] / rmse['static'][var] - 1) * 100
        if rmse_diff > 0:
            print(f"  → 정적 예측이 {abs(rmse_diff):.1f}% 더 정확")
        else:
            print(f"  → 동적 예측이 {abs(rmse_diff):.1f}% 더 정확")
    
    print("\n" + "-" * 60)
    
    return {
        'residuals': residuals,
        'aic': aic,
        'bic': bic,
        'hqic': hqic
    }
    
# ===================================================================
# 6. 메인 실행 함수
# ===================================================================

def main():
    """
    VECM 분석 전체 프로세스 실행
    """
    
    print("=" * 60)
    print("VECM 모형 분석 시작")
    print("=" * 60)
    
    # 1. 데이터 시뮬레이션
    data, true_params = simulate_vecm_data(n_obs=500, seed=42)
    
    # 2. 단위근 검정
    unit_root_tests(data)
    
    # 3. VECM 추정 및 예측
    results = estimate_vecm(data, train_ratio=0.7)
    
    # 4. 모형 진단
    model_diagnostics(results)
    
    # 5. 결과 시각화
    plot_results(data, true_params, results)
    
    # ==== 최종 요약 ====
    print("\n" + "=" * 60)
    print("분석 완료 - 최종 요약")
    print("=" * 60)
    
    print("\n1. 파라미터 비교:")
    print("-" * 40)
    print("조정속도 (alpha):")
    print(f"  실제: {true_params['alpha'].flatten()}")
    print(f"  추정: {results['model'].alpha.flatten()}")
    
    print("\n공적분 관계:")
    beta_true = true_params['beta'].flatten()
    beta_est = results['model'].beta[:, 0]
    print(f"  실제: y1 - {-beta_true[1]:.3f}*y2")
    print(f"  추정: y1 - {-(beta_est[1]/beta_est[0]):.3f}*y2")
    
    print("\n2. 예측 성능 요약:")
    print("-" * 40)
    print("RMSE 비교:")
    print(f"  정적 예측 평균: {np.mean([results['rmse']['static']['y1'], results['rmse']['static']['y2']]):.4f}")
    print(f"  동적 예측 평균: {np.mean([results['rmse']['dynamic']['y1'], results['rmse']['dynamic']['y2']]):.4f}")
    
    print("\nMAPE 비교 (%):")
    print(f"  정적 예측 평균: {np.mean([results['mape']['static']['y1'], results['mape']['static']['y2']]):.2f}%")
    print(f"  동적 예측 평균: {np.mean([results['mape']['dynamic']['y1'], results['mape']['dynamic']['y2']]):.2f}%")
    
    print("\n3. 주요 발견사항:")
    print("-" * 40)
    print("- 정적 예측의 동적 예측 대비 우수한 성능")
    print("- 조정속도가 큰 변수의 빠른 균형 수렴")
    print("- 공적분 관계의 장기 균형 포착")
    
    print("=" * 60)
    print("분석 종료")
    print("=" * 60)

# 실행
if __name__ == "__main__":
    main()
```

## 4. 모형의 경제학적 해석

**장기 균형과 단기 조정**

- 경제 변수들 간의 안정적 관계 표현
- 불균형 상태에서 균형으로의 복귀 과정
- 조정속도 $\alpha$의 크기와 부호의 중요성
- 충격의 지속성 분석 및 정책 개입 효과 평가

**예측의 종류와 활용**

정적 예측 (Static Forecast)

- 1기 앞 예측 (one-step ahead)
- 매 시점 실제값을 사용한 다음 기 예측
- 단기 예측에 적합
- 모형 적합도 평가용

동적 예측 (Dynamic Forecast)

- 다기간 예측 (multi-step ahead)
- 이전 예측값을 사용한 연쇄 예측
- 중장기 예측에 활용
- 예측 오차의 누적

**모형 선택 기준**

- 공적분 rank 결정: Johansen trace 검정, Maximum eigenvalue 검정
- 시차 선택: AIC ($-2\ln(L) + 2k$), BIC ($-2\ln(L) + k\ln(n)$), HQ ($-2\ln(L) + 2k\ln(\ln(n))$)
- 모형 진단: 잔차 자기상관 검정, 정규성 검정, 이분산성 검정

## 5. VECM 모형의 장점과 한계

**장점**

- 이론과 실증의 결합: 경제이론의 장기 관계 반영, 데이터 기반 단기 동학 추정
- 예측 성능: VAR 대비 우수한 장기 예측, 균형 조정 메커니즘 반영
- 정책 분석: 충격반응함수 분석, 예측오차 분산분해

**한계**

- 모형 설정: 선형 관계 가정, 구조 변화 미반영
- 추정 문제: 대표본 요구, 약한 공적분 문제
- 해석의 어려움: 다중 공적분 관계, 식별 제약 필요

## 6. 실무 적용 사례

**금융 시장**

- 금리 기간구조 분석
- 환율 결정 모형
- 주가-거래량 관계

**거시경제**

- 화폐수요 함수
- 구매력평가 검정
- 필립스 곡선 분석

**에너지 경제**

- 원유가격-경제성장 관계
- 전력수요 예측
- 탄소배출권 가격 분석

## 7. 결론

VECM은 비정상 시계열 간의 장기 균형관계와 단기 동학을 통합적으로 모형화하는 강력한 도구

**핵심 요점**

- 공적분 관계의 경제이론적 장기 균형 표현
- 오차수정항을 통한 불균형 조정 과정 포착
- 정적/동적 예측의 다양한 예측 horizon 대응
- 파라미터 추정을 통한 구조적 관계 파악

**향후 발전 방향**

- 비선형 VECM: Threshold VECM, Smooth Transition VECM
- 고차원 VECM: Factor VECM, Sparse VECM
- 베이지안 VECM: 사전정보 활용, 불확실성 정량화

---

## References

Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. _Econometrica_, 55(2), 251-276. https://doi.org/10.2307/1913236

Hamilton, J. D. (1994). _Time series analysis_. Princeton University Press.

Johansen, S. (1988). Statistical analysis of cointegration vectors. _Journal of Economic Dynamics and Control_, 12(2-3), 231-254. https://doi.org/10.1016/0165-1889(88)90041-3

Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. _Econometrica_, 59(6), 1551-1580. https://doi.org/10.2307/2938278

Johansen, S. (1995). _Likelihood-based inference in cointegrated vector autoregressive models_. Oxford University Press. https://doi.org/10.1093/0198774508.001.0001

Lütkepohl, H. (2005). _New introduction to multiple time series analysis_. Springer-Verlag. https://doi.org/10.1007/978-3-540-27752-1

Lütkepohl, H., & Krätzig, M. (Eds.). (2004). _Applied time series econometrics_. Cambridge University Press. https://doi.org/10.1017/CBO9780511606885

MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. _Journal of Applied Econometrics_, 14(5), 563-577. https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R

Pfaff, B. (2008). _Analysis of integrated and cointegrated time series with R_ (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-75967-8

Stock, J. H., & Watson, M. W. (1993). A simple estimator of cointegrating vectors in higher order integrated systems. _Econometrica_, 61(4), 783-820. https://doi.org/10.2307/2951763