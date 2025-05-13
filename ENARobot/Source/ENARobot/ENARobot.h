// ENARobot/Source/ENARobot/ENARobot.h

#pragma once

#include "Modules/ModuleManager.h" // 모듈 관리자 헤더 포함

// 프로젝트의 메인 모듈 클래스 선언
// F로 시작하는 이름은 UE에서 구조체(Struct)나 모듈(Module) 클래스에 자주 사용되는 컨벤션입니다.
class FENARobotModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	// 엔진이 모듈을 로드할 때 호출됩니다.
	virtual void StartupModule() override;

	// 엔진이 모듈을 언로드할 때 호출됩니다 (예: 에디터 종료, 모듈 핫 리로드).
	virtual void ShutdownModule() override;
};