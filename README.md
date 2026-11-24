# Does RFM Really Predict the Future? — 공통 코드

AI데이터분석및실습 기말 튜토리얼 프로젝트. 이 폴더는 팀 세 명이 함께 쓰는 **공통 기반(0~3단계)** 이다.

## 폴더 구조

```
rfm_utils.py                      공용 함수 (정제, 기준일 분할, 점수 형식, lift@k)
tests/test_rfm_utils.py           누수·형식 테스트 (데이터 다운로드 없이 실행)
notebooks/00_common_pipeline.ipynb 공통 파이프라인 실행과 점검
notebooks/A_eda_evaluation.ipynb   A: 정제 보고서, EDA, 평가 틀
notebooks/B_rule_rfm_tests.ipynb   B: 규칙 기반 RFM, H1·H2, K-means
notebooks/C_ml_bgnbd.ipynb         C: ML 예측, H3, BG/NBD
notebooks/final_tutorial.ipynb       제출용 글의 뼈대 (7개 장, 장마다 담당 표기). A가 조립한다
outputs/                           B·C가 저장한 점수 파일 scores_<method>.csv (커밋한다)
(ru.evaluate_scores: 모든 방법이 같이 쓰는 평가 함수 AUC·PR-AUC·lift@10·capture@10)
requirements.txt                  라이브러리 버전 (Python 3.9 이상)
data/                             원본 데이터 (자동 다운로드, 저장소에는 올리지 않음)
```

## 처음 시작하기

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
jupyter notebook notebooks/00_common_pipeline.ipynb   # jupyter가 없으면 pip install notebook
```

첫 실행에서 UCI Online Retail II(약 46MB)를 `data/`에 내려받고 gzip CSV로 캐시한다. 원본의 라이선스와 재배포 조건은 UCI 페이지에서 직접 확인한다.

## 자기 노트북에서 시작하는 법 (A·B·C 공통)

```python
import sys; sys.path.insert(0, "..")
import rfm_utils as ru

inv  = ru.invoice_table(ru.clean_transactions(ru.load_raw())[0])
data = ru.make_dataset(inv, ru.SNAPSHOTS["S2"])      # 고객 1행: R, F, M, ..., repurchase
ru.assert_no_leakage(inv, data, ru.SNAPSHOTS["S2"])  # 새 데이터셋을 만들 때마다 호출
```

## 결과 넘기기

B·C는 노트북 마지막 셀에서 `ru.save_score_frame(점수프레임, "방법이름")`으로 `outputs/scores_<방법이름>.csv`를 저장하고 커밋한다. A는 `ru.load_score_frames()`로 모두 읽어 같은 표에서 비교한다. 최종 글은 `final_tutorial.ipynb`에서 조립해 `jupyter nbconvert --to html`로 변환한다.

## 팀 약속 (바꾸지 않는다)

1. **누수 금지.** 특징은 기준일 *이전* 주문만, 정답은 기준일 *이후 90일*(`[기준일, 기준일+90일)`)만 쓴다.
2. **기준일은 `ru.SNAPSHOTS`.** S1=2011-03-01, S2=2011-06-01, S3=2011-09-01. 예측 모델은 S1 학습, S2 검증, S3 테스트.
3. **점수 형식.** 모든 방법은 `ru.to_score_frame(...)`으로 `(customer_id, snapshot, method, score)`를 낸다. 점수가 높을수록 재구매 가능성이 높다. 규칙 기반 세그먼트는 숫자 점수로 바꿔서 낸다.
4. **기준선.** `ru.baseline_recency_score(data)`("최근에 샀는가"만)보다 나아야 의미가 있다.
5. **공용 함수를 고칠 때는** `rfm_utils.py`만 고치고 `python -m pytest -q`를 통과시킨 뒤 단톡방에 알린다.
6. **라이브러리를 추가할 때는** `requirements.txt`에 버전과 함께 적는다. 랜덤 시드는 `ru.SEED`.

## 확정된 결정

- 완전 중복 행(약 3%)은 **지운다** (2026-11-17 확정, `clean_transactions`의 기본값). 같은 상품을 같은 시각에 여러 번 담은 정상 주문이 섞였을 가능성은 본문의 한계로 적는다.

- **용어: "소비자"가 아니라 "거래처 고객"** (2026-11-24, 팀 피드백 반영). Online Retail II는 도매 성격이다 (주문 중앙값 303파운드, 상품 한 줄 수량 중앙값 6개·상위 1% 144개 이상). 글과 코드 주석에서 소비자 재구매라고 쓰지 않고, 결론도 "거래처 관리(재주문 유도)"로 쓴다.
- **BG/NBD는 "가정 점검" 도전 파트.** 구매 간격 변동계수(CV) 중앙값이 0.75로 포아송(약 1)보다 규칙적이라 무작위 구매 가정이 일부 깨진다. 이 모델의 결과는 "가정이 깨진 상태에서 얼마나 버티는가"로 해석한다.
- 위 세 항목은 2026-11-24에 확정했다.
- 데이터셋은 바꾸지 않는다 (B2C 후보는 표본·재구매율·날짜 문제로 부적합, 공통 코드 재작성 비용).

## 열려 있는 결정

- 우편 요금 등 상품이 아닌 항목(StockCode가 POST 등)을 뺄지 (아직 처리하지 않음)
- 작성 언어와 라이선스 확인
