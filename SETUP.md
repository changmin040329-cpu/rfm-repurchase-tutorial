# 환경 세팅 안내 (맥·윈도우)

**목표: 11/26까지 1~5번을 끝내고 `pytest` 11개 통과 화면을 단톡방에 올린다.**

준비물: Python 3.10~3.12, git, GitHub 계정(저장소 초대 수락 완료)
버전 확인은 맥 `python3 --version`, 윈도우 `py --version`. 3.13 이상이면 3.12를 새로 설치한다.

## 1. 저장소 받기
```bash
git clone https://github.com/changmin040329-cpu/rfm-repurchase-tutorial.git
cd rfm-repurchase-tutorial
```
처음 clone할 때 GitHub 로그인 창이 뜨면 브라우저로 로그인한다.

## 2. 가상환경 만들고 켜기
**맥/리눅스**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**윈도우 (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```
켜지면 프롬프트 앞에 `(.venv)`가 보인다. **터미널을 새로 열 때마다 `activate`를 다시 실행해야 한다.** 윈도우에서 실행 정책 오류가 나면 `Set-ExecutionPolicy -Scope Process RemoteSigned` 후 다시 시도한다.

## 3. 라이브러리 설치
```bash
pip install -r requirements.txt notebook
```
몇 분 걸린다.

## 4. 환경 확인 (11개 통과하면 성공)
```bash
python -m pytest -q
```

## 5. 노트북 열기
```bash
jupyter notebook notebooks/
```
브라우저에서 자기 노트북을 연다. **A_eda_evaluation(이창민), B_rule_rfm_tests(서유나), C_ml_bgnbd(이정민).** 위에서부터 셀을 하나씩 실행한다.
- 첫 실행은 오래 걸린다. 원본 데이터 약 46MB를 자동으로 내려받는다.
- `RuntimeWarning: divide by zero` 같은 경고가 나올 수 있다. 결과에는 영향이 없다.

## 6. git 설정 (한 번만, 저장소 폴더 안에서)
```bash
git config --global push.autoSetupRemote true
pip install nbstripout && nbstripout --install
```
첫 줄은 새 브랜치를 처음 올릴 때 오류가 나지 않게 하고, 둘째 줄은 노트북을 커밋할 때 실행 결과(그림·표)를 자동으로 지워 준다.

## 7. 작업 방식 (자세한 규칙은 README의 "브랜치 전략")
```bash
git checkout main && git pull        # 작업 전에 항상 최신으로
git checkout -b b/segment-rules      # 자기 역할 접두어(a/ b/ c/) + 작업 이름
# ... 작업 후 ...
git add notebooks/B_rule_rfm_tests.ipynb
git commit -m "B: segment rules"
git push
```
그다음 GitHub에서 Pull Request를 만들고, 다른 팀원이 승인하면 병합한다. `main`에는 직접 올릴 수 없다.

## 막히면
| 증상 | 해결 |
|---|---|
| `No module named 'numpy'` 등 | `(.venv)`가 켜져 있는지 확인한다. 안 켜졌으면 2번의 activate를 실행하고 `jupyter notebook`을 다시 시작한다 |
| `pytest`가 11개보다 적게 통과 | `git pull`로 최신을 받았는지 확인한다 |
| 데이터 다운로드 실패 | 인터넷 연결을 확인하고 셀을 다시 실행한다 |
| 그 외 | 오류 메시지 전체를 단톡방에 붙여 넣는다 (하루를 넘기지 않는다) |

> 이 안내는 맥에서 검증했고, 윈도우 명령은 직접 확인하지 못했다. 윈도우에서 막히면 화면을 공유해 이 문서를 고친다.
