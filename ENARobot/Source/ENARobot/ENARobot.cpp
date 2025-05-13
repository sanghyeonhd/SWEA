// ENARobot/Source/ENARobot/ENARobot.cpp

#include "ENARobot.h" // 자체 헤더 포함
#include "Modules/ModuleManager.h" // 이 줄은 사실 ENARobot.h에 이미 포함되어 있어 필요 없을 수도 있으나, 명시적으로 포함하는 경우도 있습니다.

// UE 로깅 매크로에 사용할 네임스페이스 정의 (선택 사항이지만 흔히 사용)
#define LOCTEXT_NAMESPACE "FENARobotModule"

// StartupModule 구현: 모듈 로드 시 실행
void FENARobotModule::StartupModule()
{
	// 이 코드는 모듈이 메모리에 로드된 후 실행됩니다.
	// .uplugin 파일에 로딩 시점이 명시될 수 있습니다.
	UE_LOG(LogTemp, Log, TEXT("FENARobotModule module starting up.")); // 로그 출력 예시

	// 플러그인이나 모듈 초기화 관련 추가 코드를 여기에 작성할 수 있습니다.
}

// ShutdownModule 구현: 모듈 언로드 시 실행
void FENARobotModule::ShutdownModule()
{
	// 이 함수는 종료 시 모듈 정리 작업을 위해 호출될 수 있습니다.
	UE_LOG(LogTemp, Log, TEXT("FENARobotModule module shutting down.")); // 로그 출력 예시

	// 모듈 정리 관련 추가 코드를 여기에 작성할 수 있습니다.
}

#undef LOCTEXT_NAMESPACE // 정의된 네임스페이스 해제

// IMPLEMENT_MODULE 매크로: 모듈을 UE 시스템에 등록합니다.
// 첫 번째 인자는 모듈 클래스 이름, 두 번째 인자는 모듈 이름 (Build.cs에서 정의한)
IMPLEMENT_MODULE(FENARobotModule, ENARobot);