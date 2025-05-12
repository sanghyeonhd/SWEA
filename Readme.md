# SWEA XXXXX 프로젝트

실시간 센서 데이터, AI 예측, 물리 시뮬레이션, 로봇 제어 등을 통합하여 용접 공정의 품질을 최적화하고 적응 제어를 수행하기 위한 디지털트윈 AI 시스템의 초기 개발 프로젝트입니다.

## 프로젝트 배경 및 범위

이 프로젝트는 용접 공정의 디지털 트윈을 구축하여 실시간 데이터 분석 및 AI 기반 의사결정을 통해 용접 품질을 관리하고 최적화하는 것을 목표로 합니다. 현재 구현 상태 및 포함 범위는 기본적인 구조와 핵심 모듈의 뼈대를 제공하는 수준이며, 실제 현장 적용을 위해서는 많은 추가 개발이 필요합니다.

*   **Unreal Engine 연동:** `physics_interface.py`는 Python 코드와 Unreal Engine 시뮬레이터 간의 통신 인터페이스를 개념적으로 정의합니다. 실제 UE 프로젝트 내 물리 시뮬레이션 로직 구현 및 Python과의 통신 프로토콜 구현은 별도의 UE 개발 작업이 필요합니다.
*   **AI 모델 (PyTorch):** AI 모델 아키텍처(`ai_model.py`)는 정의되어 있으나, 실제 학습을 위한 대량의 레이블링된 데이터와 학습 실행 (`trainer.py`)을 통한 모델 가중치 파일(`welding_model.pth`) 및 스케일러 파일(`scaler.pkl`) 생성이 선행되어야 합니다. 예제 코드는 모델 로딩 및 추론 구조를 보여줍니다.
*   **로봇 제어 (현대 로보틱스):** `robot_control_interface.py`는 현대로보틱스 로봇 컨트롤러와의 통신 인터페이스를 가상 프로토콜 기반으로 정의합니다. 실제 로봇 컨트롤러의 통신 프로토콜 구현 및 연동 작업이 필요합니다.
*   **데이터베이스:** `data_logger_db.py`는 SQLite를 사용한 기본적인 로깅 및 데이터 관리 기능을 개념적으로 구현합니다. 실제 현장에서는 요구사항에 맞는 데이터베이스 시스템(PostgreSQL, InfluxDB 등) 선택 및 상세 스키마 설계, 연동 구현이 필요합니다.
*   **실시간 데이터 수집:** `sensor_data_handler.py`는 실시간 센서 데이터 수집 및 처리를 위한 인터페이스로, 구체적인 센서별 통신/수집 로직 구현이 필요합니다.
*   **완전한 기능:** 이 코드는 이미지에 명시된 전체 시스템의 기본 뼈대를 제공하며, 각 모듈이 어떻게 구성될 수 있는지를 보여줍니다. 실제 현장에 적용하기 위해서는 많은 추가 개발과 데이터 기반의 상세 구현이 필요합니다.

## 파일 구조

프로젝트의 주요 파일 구조는 다음과 같습니다. (`src/` 디렉토리 아래에 소스 코드가 위치하는 구조를 제안합니다.)

```text
WeldingDigitalTwinAI/
├── .gitattributes
├── .gitignore
├── README.md               # 프로젝트 개요 및 설명 (현재 파일)
├── requirements.txt        # Python 의존성 라이브러리 목록
├── setup.py                # 프로젝트 패키징 및 설치 설정
└── src/                    # 메인 소스 코드 디렉토리
    ├── __init__.py         # src 디렉토리를 Python 패키지로 만듦 (최소한의 내용)
    ├── config.py           # 시스템 설정 값 관리
    ├── data_handler.py     # AI 학습을 위한 과거/배치 데이터 처리 (CSV 예시)
    ├── sensor_data_handler.py # 실시간 센서 데이터 수집 및 처리 (개념적)
    ├── ai_model.py         # PyTorch 기반 AI 모델 아키텍처 정의
    ├── trainer.py          # AI 모델 학습 로직 및 모델/스케일러 저장 기능 포함
    ├── predictor.py        # 용접 결과 예측 로직 (AI/물리 기반 - 초기 뼈대)
    ├── ai_inference_engine.py # 학습된 AI 모델을 사용한 실시간 추론 엔진
    ├── evaluator.py        # 예측 결과에 대한 기본적인 품질 평가 (초기 뼈대)
    ├── quality_evaluator_adaptive_control.py # 종합 품질 평가 및 적응 제어 결정 로직
    ├── physics_interface.py # Unreal Engine 물리 시뮬레이션 연동 인터페이스 (가상)
    ├── robot_control_interface.py # 로봇 컨트롤러 연동 인터페이스 (가상)
    ├── welding_process_manager.py # 전체 용접 작업 흐름 관리 및 장비 조율
    ├── data_logger_db.py   # 데이터베이스 로깅 및 데이터 관리 (SQLite 예시)
    └── system_manager.py   # 전체 시스템 모듈 초기화, 시작, 중지 및 조율 (메인 컨트롤러)
├── data/                   # 예시 데이터 파일 저장 디렉토리
    ├── dummy_sensor_data.csv # 예시 센서 데이터 (내용상 `labels.csv`로 사용될 가능성 있음)
    └── dummy_labels.csv      # 예시 레이블 데이터 (내용상 `sensor_data.csv`로 사용될 가능성 있음)
├── models/                 # 학습된 모델 및 관련 파일 저장 디렉토리
    ├── welding_model.pth   # 학습된 AI 모델 가중치 파일 (trainer.py 실행으로 생성 필요)
    └── scaler.pkl          # 학습 시 사용된 데이터 스케일러 객체 파일 (trainer.py 실행으로 생성 필요)
# └── hmi_application/        # 사용자 인터페이스 파일 (별도 애플리케이션 또는 레포지토리 가능성 높음)

### 주요 파일 설명

*   **`src/config.py`**: 시스템 전반에 사용되는 설정 값들을 관리합니다. 로봇 IP/포트, DB 경로, 모델 경로, AI 파라미터 범위, 품질 클래스 등 시스템의 모든 구성 요소를 위한 설정이 포함됩니다.
*   **`src/data_handler.py`**: AI 모델 학습을 위한 과거/배치 데이터(주로 CSV)를 로딩, 병합, 전처리(스케일링 포함)하고 PyTorch Dataset/DataLoader 형태로 준비합니다. 이미지 데이터 처리 기능도 포함될 수 있습니다.
*   **`src/sensor_data_handler.py`**: (개념적) 다양한 센서(용접기 피드백, 비전, 레이저 등)로부터 실시간 데이터 스트림을 수집하고 처리하는 역할을 합니다. 수집된 데이터를 동기화하고 필요한 전처리를 거쳐 다른 모듈(AI 엔진, 품질 평가기)에 전달합니다. **구체적인 센서별 통신/수집 로직 구현이 필요합니다.**
*   **`src/ai_model.py`**: PyTorch 기반 딥러닝 모델의 신경망 아키텍처를 정의합니다. 센서 데이터 및 선택적으로 이미지 데이터를 입력받아 용접 품질 관련 예측을 수행합니다.
*   **`src/trainer.py`**: `data_handler`의 데이터를 사용하여 `ai_model`의 인스턴스를 학습시키는 핵심 로직을 포함합니다. 학습 완료 후 최적 모델 가중치(`welding_model.pth`) 및 입력 스케일러 객체(`scaler.pkl`)를 `models/` 디렉토리에 저장합니다. **이미지 데이터 학습 및 회귀 문제 지원 로직 보완이 필요합니다.**
*   **`src/predictor.py`**: (초기 뼈대) AI 모델 또는 물리 시뮬레이션을 사용하여 단일 용접 결과 예측 요청을 처리하는 로직입니다. `ai_inference_engine.py`가 실시간 추론을 담당함에 따라 역할 재정의가 필요합니다.
*   **`src/ai_inference_engine.py`**: 학습된 AI 모델 및 스케일러를 로드하여 `sensor_data_handler`로부터 오는 **실시간 센서 데이터**에 대한 추론을 수행합니다. 예측 결과(품질 클래스, 점수, 신뢰도 등)를 `quality_evaluator_adaptive_control.py` 등으로 전달합니다. **모델 및 스케일러 로딩 견고화, 이미지 추론 지원 보완이 필요합니다.**
*   **`src/evaluator.py`**: (초기 뼈대) AI 예측 또는 물리 시뮬레이션 결과에 대한 기본적인 품질 평가를 수행합니다. `quality_evaluator_adaptive_control.py`로 기능이 확장되거나 통합될 수 있습니다.
*   **`src/quality_evaluator_adaptive_control.py`**: `ai_inference_engine`의 예측 결과, `sensor_data_handler`의 실시간 센서 데이터, (선택적으로) 물리 시뮬레이션 결과를 종합하여 용접 품질의 최종 상태를 평가합니다. 평가 결과에 기반하여 미리 정의된 규칙이나 정책에 따라 실시간 적응 제어 파라미터 조정을 결정하고 제안합니다. **실제 공정 지식 기반의 정교하고 상세한 적응 제어 규칙 및 로직 구현이 가장 중요합니다.**
*   **`src/physics_interface.py`**: Unreal Engine 기반 물리 시뮬레이션과의 TCP/IP 소켓 통신 인터페이스를 정의합니다. 용접 파라미터를 시뮬레이터에 보내고 시뮬레이션 결과(예: 비드 형상, 품질 점수)를 받아오는 역할을 합니다 (가상 프로토콜). **UE 측과의 실제 통신 프로토콜 상세 구현 및 견고한 통신/오류 처리 로직 구현이 필요합니다.**
*   **`src/robot_control_interface.py`**: 현대로보틱스 로봇 컨트롤러와의 TCP/IP 소켓 통신 인터페이스를 정의합니다. 로봇 상태 모니터링, 프로그램 실행, 파라미터 설정, I/O 제어 등 로봇 제어 명령을 전송하고 응답을 처리합니다 (가상 프로토콜). 여러 대의 로봇 통신 관리를 포함합니다. **실제 로봇 컨트롤러 통신 프로토콜 상세 구현, 비동기 처리, 오류 처리, 안전 로직 구현이 필수적입니다.**
*   **`src/welding_process_manager.py`**: 전체 용접 작업 시퀀스(레시피 기반)를 관리하고, `robot_control_interface`, `sensor_data_handler`, `ai_inference_engine`, `quality_evaluator_adaptive_control` 등 핵심 모듈 간의 작업을 조율합니다. 적응 제어 루프를 관리하고 로봇에게 조정 명령을 전달합니다. **실제 작업 레시피 관리 시스템 및 복잡한 시퀀싱, 오류 복구 로직 구현이 필요합니다.**
*   **`src/data_logger_db.py`**: SQLite 데이터베이스를 사용하여 시스템에서 발생하는 모든 종류의 데이터(센서, 예측, 평가, 제어 행동, 로봇 상태, 공정 이벤트 등)를 체계적으로 기록하고 관리합니다. **실제 DB 시스템(PostgreSQL, TimescaleDB 등) 선택 및 연동 구현, 스키마 상세화, 대용량/고속 로깅 처리 구현이 필요합니다.**
*   **`src/system_manager.py`**: 시스템의 최상위 컨트롤러이자 진입점입니다. 모든 핵심 모듈을 초기화, 시작, 중지하며 생명주기를 관리합니다. 시스템 상태 모니터링 및 외부 명령 처리 인터페이스를 포함합니다. **모듈 상태 모니터링/복구, 외부 연동 인터페이스(HMI, API) 구현이 필요합니다.**
*   **`README.md`**: 프로젝트의 목적, 구조, 설정, 사용법, 개발 현황 등을 설명하는 문서입니다 (현재 보고 계신 파일).
*   **`requirements.txt`**: 이 프로젝트를 실행하기 위해 `pip`으로 설치해야 하는 모든 Python 라이브러리 목록을 정의합니다. 새로 추가될 모듈들에 필요한 라이브러리도 포함하여 업데이트해야 합니다.
*   **`setup.py`**: `setuptools`를 사용하여 프로젝트를 Python 패키지로 만들고 설치하는 방법을 정의하는 스크립트입니다. 의존성, 진입점(console scripts) 등을 명시합니다.
*   **`src/__init__.py`**: `src` 디렉토리를 유효한 Python 패키지로 인식하도록 합니다. 최소한의 내용만 포함합니다.
*   **`data/`**: 학습 또는 테스트를 위한 데이터 파일이 저장될 디렉토리입니다. `dummy_sensor_data.csv`, `dummy_labels.csv`와 같은 예시 파일이 포함됩니다. **(참고: 현재 제공된 CSV 파일의 내용과 파일 이름이 서로 바뀌어 있는 것으로 보입니다. 내용상 `dummy_sensor_data.csv`는 센서 데이터가, `dummy_labels.csv`는 레이블/점수 데이터가 저장되어야 합니다.)**
*   **`models/`**: 학습된 AI 모델 가중치 파일(`welding_model.pth`) 및 데이터 스케일링에 사용된 스케일러 객체 파일(`scaler.pkl`)이 저장될 디렉토리입니다. `trainer.py`에 의해 생성되고 `ai_inference_engine.py` 등에 의해 로드됩니다. **실제 학습 실행을 통해 생성해야 합니다.**
*   **`hmi_application/`**: (개념적) 사용자 인터페이스(Human-Machine Interface) 관련 파일들이 포함될 디렉토리입니다. 운영자가 시스템을 모니터링하고 제어하는 기능을 제공하며, 별도 애플리케이션으로 개발될 수 있습니다.

## 설치 방법

프로젝트를 실행하기 위해 필요한 라이브러리는 `requirements.txt`에 명시되어 있습니다. Python 3.8 이상 환경에서 다음 명령을 사용하여 설치할 수 있습니다.

```bash
# (선택 사항) 가상 환경 생성 및 활성화 - 프로젝트 환경 분리를 위해 권장
# python -m venv .venv
# source .venv/bin/activate # Linux/macOS
# .venv\Scripts\activate # Windows

# 의존성 라이브러리 설치
pip install -r requirements.txt

# 프로젝트를 패키지로 설치 (개발 중에는 소스 코드 수정 시 자동 반영되는 개발 모드 권장)
# setup.py의 PACKAGE_DIR = {'': 'src'} 설정에 따라 소스 코드가 'src/' 디렉토리 아래에 있다고 가정
pip install -e .
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
실행 방법

src/config.py 파일에 필요한 설정을 확인하고 실제 환경에 맞게 수정합니다 (로봇 IP/포트, DB 경로, 모델/스케일러 파일 경로 등).

(필요시) trainer.py를 실행하여 AI 모델을 학습시키고 models/welding_model.pth 및 models/scaler.pkl 파일을 생성합니다. 학습 데이터가 없을 경우, 테스트를 위해 더미 파일을 생성하는 별도 스크립트나 방법을 사용할 수 있습니다.

(선택 사항) physics_interface.py, robot_control_interface.py를 테스트하거나 사용하려면 해당 가상 프로토콜과 포트에 맞게 동작하는 가상 서버 또는 실제 장비를 준비하고 실행합니다.

프로젝트 루트 디렉토리 (README.md 파일이 있는 곳)에서 다음 명령을 실행하여 시스템 관리자(system_manager.py)를 시작합니다.

# setup.py의 ENTRY_POINTS 설정에 따라 설치된 콘솔 스크립트 실행
samsung_welding_dt_start

# 또는 setup.py가 없거나 ENTRY_POINTS를 사용하지 않는 경우, 직접 실행 (src/ 구조 가정)
# python -m src.system_manager
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

시스템이 시작되면 system_manager는 config.py를 읽어 모듈들을 초기화하고, 로봇 연결, 센서 데이터 수집 시작 등의 초기화 과정을 거쳐 대기 상태가 됩니다. welding_process_manager.py의 start_welding_job 등의 메서드를 외부(예: HMI 연동 모듈)에서 호출하거나, system_manager가 제공하는 외부 제어 인터페이스를 통해 특정 용접 작업을 시작할 수 있습니다. Ctrl+C 또는 시스템 종료 신호(SIGINT, SIGTERM)로 시스템을 정상 종료할 수 있습니다.

추가 개발 및 개선 사항 (Work in Progress)

이 프로젝트는 현재 기본적인 구조를 갖추었으며, 실제 현장 적용을 위해서는 다음과 같은 추가 개발 및 개선이 필요합니다.

설정 관리: config.py 상세화 및 중앙 집중식 관리 강화. 복잡한 설정(레시피, 제어 규칙) 관리 시스템 구축. 실행 중 동적 설정 변경 기능.

데이터 처리 파이프라인:

data_handler.py: 이미지 데이터 처리, Scaler 저장/로딩 로직 완성.

sensor_data_handler.py: 다양한 센서와의 실제 통신 드라이버 구현, 실시간 데이터 수집, 동기화, 전처리, 효율적인 데이터 분배(메시지 큐 등) 로직 구축.

AI/ML:

ai_model.py: 시계열, 이미지 데이터 처리를 위한 고급 아키텍처 도입 및 최적화.

trainer.py: 이미지 학습, 회귀 지원, 학습 가시화/로깅, 조기 종료, 하이퍼파라미터 튜닝 통합. Scaler 저장 로직 구현.

ai_inference_engine.py: 모델/스케일러 로딩 견고화, 이미지 추론 지원 완성, 실시간 스트림 추론 성능 최적화.

품질 평가 및 제어:

evaluator.py / quality_evaluator_adaptive_control.py: 실제 공정 지식 기반의 정교하고 상세한 적응 제어 규칙 및 제어 로직 구현. 다중 평가 소스 종합, 제어 파라미터 조정 계산 정밀화, 안전한 제한 적용 강화.

인터페이스:

physics_interface.py: Unreal Engine 측과의 실제 통신 프로토콜 상세 구현 및 견고화. 연결/오류 관리.

robot_control_interface.py: 현대로보틱스 컨트롤러와의 실제 통신 프로토콜 상세 구현 및 견고화. 비동기 처리, 명령 큐, 오류 처리, 안전 로직, 다중 로봇 조율 기능 강화.

공정 관리: welding_process_manager.py: 실제 작업 레시피 관리 시스템 구축. 복잡한 작업 시퀀싱, 동기화, 오류 복구 절차 상세 구현. 적응 제어 루프와의 정교한 연동.

데이터 관리: data_logger_db.py: 요구사항에 맞는 실제 DB 시스템(PostgreSQL, TimescaleDB 등) 선택 및 연동 구현. 스키마 상세화 및 최적화. 대용량/고속 로깅 처리, 데이터 조회/분석 인터페이스 개발.

시스템 관리: system_manager.py: 모듈 상태 모니터링/자동 복구, 외부 명령 수신 인터페이스(HMI, API), 보안 기능 통합.

사용자 인터페이스 (HMI): 별도 애플리케이션 개발. 시스템 상태 시각화, 작업 설정/관리, 실시간 데이터/결과 표시, 알람, 수동 제어 기능 구현. 시스템 관리자와의 통신 인터페이스 구축.

보안: 시스템 전반에 걸친 인증, 권한 부여, 데이터 암호화 등 보안 메커니즘 구현.

라이선스

이 프로젝트는 삼성 E&A의 사내 개발 프로젝트이며, 라이선스는 'Proprietary' 입니다. 외부 배포 및 사용은 제한됩니다. (필요에 따라 변경)

**위의 마크다운 코드를 복사하여 `README.md` 파일로 저장하시면 됩니다.**

이 업데이트된 `README.md`는 현재까지 논의된 시스템의 모든 핵심 구성 요소, 파일 구조, 그리고 앞으로 나아가야 할 구체적인 개발 방향을 상세히 담고 있습니다.

이제 이 `README.md`를 시스템 개발의 가이드라인으로 활용하여 각 모듈의 상세 설계 및 구현 작업을 진행할 수 있습니다. 어떤 부분부터 실제 코딩 또는 상세 설계에 착수할지 알려주시면 됩니다.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END