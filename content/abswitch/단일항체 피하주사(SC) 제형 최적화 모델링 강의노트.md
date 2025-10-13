

> **주제:** IV(정맥주사) → SC(피하주사) 제형 전환을 위한 모델링 프레임워크  
> **목적:** 항체치료제의 약동학(PK) 기반 SC 제형 최적화 데이터 구축 및 예측모형 설계  
> **구성:** ① SC 개발 성과 정리 ② 최소 변수 세트(MVD) ③ 텍스트 임베딩 변수 ④ 실습 및 과제

## 1️⃣ 단일항체 피하주사(SC) 제형 개발 성과 요약 (2025 기준)

|SC명|IV명|성분명|성과|
|---|---|---|---|
|Herceptin Hylecta / Herceptin SC|Herceptin|Trastuzumab|✅ 성공|
|Darzalex Faspro|Darzalex|Daratumumab|✅ 성공|
|Rituxan Hycela / MabThera SC|Rituxan / MabThera|Rituximab|✅ 성공|
|Kesimpta|Arzerra|Ofatumumab|✅ 성공|
|Xolair SC|Xolair (초기 IV 버전)|Omalizumab|✅ 성공|
|Nucala SC|Nucala (IV)|Mepolizumab|✅ 성공|
|Fasenra SC|Fasenra (IV)|Benralizumab|✅ 성공|
|Lecanemab SC|Leqembi (IV)|Lecanemab|✅ 성공|
|Eculizumab SC / Ravulizumab SC|Soliris / Ultomiris|Eculizumab|✅ 성공|
|Entyvio SC|Entyvio (IV)|Vedolizumab|✅ 성공|
|Keytruda Qlex (SC)|Keytruda|Pembrolizumab|⚠️ 지체 (허가 진행 중)|
|Opdivo Qvantig (SC)|Opdivo|Nivolumab|⚠️ 지체 (허가 대기)|
|Tecentriq Hybreza (SC)|Tecentriq|Atezolizumab|⚠️ 지체 (임상 종료 → 허가 진행 중)|
|Eptinezumab SC|Vyepti|Eptinezumab|⚠️ 지체 (1상 진행 중)|
|Teprotumumab SC|Tepezza (IV)|Teprotumumab|⚠️ 지체 (점도 문제로 지연)|
|Durvalumab SC|Imfinzi|Durvalumab|⚠️ 지체 (임상 진행 느림)|
|Remicade SC / CT-P13 SC|Remicade (IV)|Infliximab|⚠️ 지체 (국가별 허가 지연)|
|Actemra SC (Pen)|Actemra (IV)|Tocilizumab|❌ 실패 (대량 치료용 전환 불가)|
|Bevacizumab SC|Avastin|Bevacizumab|❌ 실패 (고점도·주입불가)|
|Cetuximab SC|Erbitux|Cetuximab|❌ 실패 (흡수율 불균일·면역반응)|
|Alemtuzumab SC|Lemtrada|Alemtuzumab|❌ 실패 (피하 흡수 불량·부작용)|
|Ipilimumab SC|Yervoy|Ipilimumab|❌ 실패 (PK 불량 및 독성 증가)|

💡 _성공 10종 / 지체 7종 / 실패 5종_  
→ SC 성공의 핵심은 **고농축 제형 안정화 + hyaluronidase 병용 + F(생체이용률) 확보**

---

## 2️⃣ 최소 변수 세트 (MVD – 20 Variables)

|구분|변수명|단위|변수유형|설명|
|---|---|---|---|---|
|**Formulation**|conc_mg_per_ml|mg/mL|수치형|항체 농도|
||viscosity_cp|cP|수치형|점도 (투여 가능성 제한)|
||inj_volume_ml|mL|수치형|주입 부피|
||pH|–|수치형|안정성·흡수성 관련|
||additive_type|–|범주형|질소, 만니톨 등 첨가제|
||enzyme_ratio|%|수치형|항체 대비 효소비율|
|**PK – IV**|CL_IV|mL/h/kg|수치형|IV 청소율|
||Vd_IV|L/kg|수치형|분포용적|
||t_half_IV|h|수치형|반감기|
||AUC_IV|μg·h/mL|수치형|기준 노출량|
||Cmax_IV|μg/mL|수치형|최대농도|
|**PK – SC (target)**|Ka_SC|h⁻¹|수치형|흡수속도상수|
||F_SC|%|수치형|생체이용률 (AUC_SC / AUC_IV × 100)|
||AUC_SC|μg·h/mL|수치형|SC 노출량|
|**Antibody Descriptor**|MW|kDa|수치형|분자량|
||pI|–|수치형|등전점|
||glyco_content|%|수치형|당화율|
|**Administration**|inj_rate_ml_min|mL/min|수치형|주입속도|
||injection_site|–|범주형|복부, 대퇴부 등|
||BMI|kg/m²|수치형|환자 생리적 흡수 차 반영|

💡 대부분 공개 문헌, FDA/EMA 라벨, 임상시험 결과로 확보 가능  
→ **예측 목표(Target)** : `Ka_SC`, `F_SC`  
→ **학습 입력(Features)** : Formulation + IV PK + Antibody 특성 + Administration

---

## 3️⃣ 텍스트 기반 임베딩 변수 (연구의 차별점)

|카테고리|변수명|형태|설명|활용 예시|
|---|---|---|---|---|
|라벨 문구 (Label-based)|`label_summary_text`|Text|FDA/EMA 허가서의 PK Summary·투여 조건 문장|“A 600 mg SC dose provides equivalent exposure to 8 mg/kg IV.”|
|논문·특허 기술요약|`formulation_description`|Text|제형 조성·첨가제·기술 요약|“Co-formulated with recombinant hyaluronidase to reduce viscosity.”|
|흡수 메커니즘 요약|`absorption_mechanism`|Text|diffusion / lymphatic pathway 관련 서술|“Absorption mainly via lymphatic drainage.”|
|안전성 문맥|`safety_text`|Text|injection-site reaction, ADA 문구|“Local reactions were mild and transient.”|
|제조사 개발노트 / 특허요약|`tech_background`|Text|제조공정·농축 기술 관련 문장|“High-concentration formulation stabilized by histidine buffer.”|

💡 _임베딩 모델 예시:_ `text-embedding-3-large` (OpenAI) or `BioBERT`  
→ 텍스트 벡터를 수치변수와 결합하여 “문헌 기반 제형지식 + PK 데이터” 융합 예측 가능

## 4️⃣ 발표 내용 (Assignments)

### 🧠 **데이터 구축**

- 항체 3종(성공, 지체, 실패)을 선택하여, IV–SC 전환 데이터를 수집하시오.
        
- 각 항체별로 최소 변수(MVD-20)를 문헌에서 추출(프롬프트로 해결할 수 있는지 가능성 타진)
    
- 출처(FDA label, PubMed, ClinicalTrials.gov 링크 포함) 기록
    

### 💻 **텍스트 임베딩 생성**

- 허가 문서 또는 논문 초록에서 임베딩 변수값을 추출하고 임베딩 벡터를 생성하시오.
         
- `text-embedding-3-large` 사용