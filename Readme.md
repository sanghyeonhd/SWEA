삼X이XX이 Wellding Engine AI


1. Unreal Engine 연동: 이 코드만으로는 Unreal Engine과의 직접적인 연동이 이루어지지 않습니다. Unreal Engine 프로젝트 설정, C++/블루프린트 스크립팅, 그리고 Python과의 통신 (예: TCP/IP, gRPC, OSC 등)을 위한 별도의 작업이 필요합니다. 여기서는 Python 측면에서의 인터페이스 역할만 가정합니다.
2. AI 모델 학습: 제공된 코드는 PyTorch 모델의 구조를 정의하지만, 실제 학습을 위해서는 대량의 레이블링된 데이터 (센서 데이터, 비드 형상 이미지 등)가 필요합니다. 데이터가 없으므로 학습된 모델 파일을 제공할 수는 없습니다.
3. 물리 시뮬레이션: Unreal Engine 기반 물리 시뮬레이션 로직은 Python 코드가 아닌 Unreal Engine 내부에 구현되어야 합니다.
4. 데이터베이스: 실제 데이터베이스 연동 코드는 포함되지 않았습니다. 데이터 로딩 부분은 예시 파일(CSV)을 읽는 형태로 구현합니다.
5. 완전한 기능: 이 코드는 이미지에 명시된 전체 시스템의 기본 뼈대를 제공하며, 각 모듈이 어떻게 구성될 수 있는지를 보여줍니다. 실제 현장에 적용하기 위해서는 많은 추가 개발과 데이터 기반의 상세 구현이 필요합니다.


파일 목록:

config.py: 설정 값 (파라미터 범위, 파일 경로 등) 관리
data_handler.py: 데이터 로딩, 전처리, DB 연동 (여기서는 CSV 예시) 담당
physics_interface.py: Unreal Engine 물리 시뮬레이션과의 연동 인터페이스 (가상)
ai_model.py: PyTorch 기반 AI 모델 구조 정의
trainer.py: AI 모델 학습 로직
predictor.py: 용접 결과 예측 로직 (AI 또는 물리 시뮬레이션 기반)
evaluator.py: 예측 결과 평가 및 품질 점수 계산
main.py: 전체 프로세스 실행 스크립트
requirements.txt: 필요한 파이썬 라이브러리 목록
dummy_sensor_data.csv: 예시 센서 데이터 파일
dummy_labels.csv: 예시 레이블 데이터 파일
