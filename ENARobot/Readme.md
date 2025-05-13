네, 알겠습니다. 삼성 E&A 디지털트윈 AI 시스템 전문가로서, Unreal Engine 프로젝트 "ENARobot"에서 Python 시스템과의 연동 및 로봇 에셋 시각화를 위해 구성 및 개발해야 할 모든 사항들을 종합하여 하나의 Markdown 형식 문서로 작성해 드리겠습니다.

이 문서는 Unreal Engine 프로젝트의 구조, 각 파일의 역할, 주요 C++ 클래스의 개념, 필수 Blueprint 에셋의 구성 방법, 그리고 Python 시스템과의 연동을 위해 남은 구체적인 개발 작업 목록을 포함합니다.

# ENARobot: Unreal Engine 디지털트윈 시각화

이 Unreal Engine 프로젝트는 Python 기반 디지털트윈 용접 엔진 AI 시스템의 시각화 서버 역할을 수행합니다. Python 시스템으로부터 로봇의 실시간 자세 및 용접 상태 데이터를 수신하여 3D 환경에 구현된 로봇 에셋의 움직임과 용접 시각 효과를 실시간으로 반영하는 것을 목표로 합니다.

## 프로젝트 배경 및 범위

이 프로젝트는 Python 시스템(클라이언트)과 네트워크 통신을 통해 데이터를 주고받으며, 수신된 데이터를 기반으로 UE 환경 내 3D 로봇 모델을 제어하고 시각적인 용접 효과를 표현합니다. 또한, Python 시스템의 요청에 따라 UE 내에서 물리 시뮬레이션을 실행하거나 시각 데이터를 반환하는 역할도 포함할 수 있습니다.

*   **Python 시스템 연동:** Python의 `physics_interface.py`와 TCP/IP 소켓 통신을 수행합니다. Python에서 정의된 가상 프로토콜(4바이트 길이 + JSON)을 사용하여 메시지를 주고받습니다.
*   **로봇 시각화:** Python에서 수신된 관절 각도 또는 TCP 트랜스폼 데이터를 사용하여 로봇 스켈레탈 메시의 본(Bone) 트랜스폼을 실시간으로 업데이트하여 로봇 움직임을 시각화합니다.
*   **용접 시각 효과:** Python에서 수신된 용접 상태 명령(아크 On/Off 등) 및 파라미터(전류, 전압 등)를 기반으로 파티클 시스템, 라이트 등을 제어하여 용접 아크, 스패터, 용접 비드 등을 시각적으로 표현합니다.
*   **물리 시뮬레이션 (선택 사항):** Python 요청에 따라 UE 내에서 용접 공정의 물리적 측면을 시뮬레이션하고 결과를 Python으로 반환합니다.
*   **시각 데이터 반환 (선택 사항):** 시뮬레이션된 카메라 뷰 이미지 등의 시각 데이터를 Python으로 반환합니다.

## Unreal Engine 프로젝트 파일 구조

"ENARobot" 프로젝트의 주요 파일 구조는 다음과 같습니다.

```text
ENARobot/  # 언리얼 엔진 프로젝트 루트 디렉토리
├── Config/             # 언리얼 엔진 설정 파일 디렉토리
│   ├── DefaultEngine.ini   # 엔진 기본 설정 (네트워킹 설정 포함 가능)
│   ├── DefaultGame.ini     # 게임 플레이 기본 설정 (시작 맵 등)
│   └── DefaultEditor.ini   # 에디터 설정 (프로젝트 생성 시 포함)
│   └── DefaultInput.ini    # 입력 설정 (프로젝트 생성 시 포함)
│   └── Default*.ini        # 기타 설정 파일
├── Content/            # 게임 에셋 디렉토리 (블루프린트, 메시, 맵, 파티클 등)
│   ├── Maps/               # 맵 파일 디렉토리
│   │   └── YourLevel.umap      # 로봇과 시각화 컴포넌트가 배치될 맵 파일 (에디터에서 생성)
│   ├── Blueprints/         # 블루프린트 에셋 디렉토리
│   │   ├── BP_DigitalTwinManager # UPythonCommServerComponent를 가질 액터 블루프린트 (에디터에서 생성)
│   │   ├── BP_MyRobotActor # ARobotBaseActor를 상속받는 로봇 액터 블루프린트 (에디터에서 생성)
│   │   └── ... (Other Blueprints for scene management, UI, etc.)
│   ├── Meshes/             # 3D 메시 에셋 (로봇 스켈레탈 메시, 용접 팁, 부품 모델 등 - 임포트)
│   ├── Materials/          # 머티리얼 에셋 (임포트 또는 에디터에서 생성)
│   ├── Particles/          # 파티클 에셋 (용접 아크, 스패터 등 - 임포트 또는 에디터에서 생성)
│   ├── Animations/         # 애니메이션 에셋 (로봇 애니메이션, 애니메이션 블루프린트 등 - 임포트/에디터 생성)
│   ├── DataTables/         # 데이터 테이블 (필요시, 예: 로봇 ID별 설정)
│   └── ... (Other Assets like sounds, textures, UI widgets)
├── Source/             # C++ 소스 코드 디렉토리
│   ├── ENARobot/ # 메인 게임 모듈 디렉토리 (프로젝트 이름과 동일)
│   │   ├── ENARobot.Build.cs     # 메인 모듈 빌드 설정
│   │   ├── ENARobot.h            # 메인 모듈 헤더 파일
│   │   ├── ENARobot.cpp          # 메인 모듈 소스 파일
│   │   ├── RobotBaseActor.h      # 로봇 액터 C++ 기본 클래스 헤더 파일
│   │   └── RobotBaseActor.cpp    # 로봇 액터 C++ 기본 클래스 소스 파일
│   └── ... (Other Game Modules if any)
└── Plugins/            # 플러그인 디렉토리
    └── DigitalTwinCommPlugin/ # Python 통신 및 데이터 처리를 위한 커스텀 플러그인 디렉토리
        ├── DigitalTwinCommPlugin.uplugin # 플러그인 정의 파일
        ├── Source/             # 플러그인 소스 코드 디렉토리
        │   └── DigitalTwinCommPlugin/ # 플러그인 모듈 디렉토리 (플러그인 이름과 동일)
        │       ├── DigitalTwinCommPlugin.Build.cs # 플러그인 모듈 빌드 설정
        │       ├── DigitalTwinCommPlugin.h        # 플러그인 모듈 헤더 파일
        │       └── DigitalTwinCommPlugin.cpp      # 플러그인 모듈 소스 파일
        │       ├── PythonCommServerComponent.h    # 통신 서버 컴포넌트 C++ 클래스 헤더 파일
        │       └── PythonCommServerComponent.cpp  # 통신 서버 컴포넌트 C++ 클래스 소스 파일
        └── ... (Other Plugin content like Shaders, Assets if any)

주요 파일 및 구성 요소 설명
1. 프로젝트 및 기본 설정 파일

ENARobot.uproject (ENARobot/ENARobot.uproject)

역할: 언리얼 엔진 프로젝트의 루트 정의 파일. 로드할 게임 모듈(ENARobot) 및 플러그인(DigitalTwinCommPlugin)을 명시합니다.

코드: (UE 에디터 생성)

{
	"FileVersion": 3,
	"EngineAssociation": "...", // 사용하는 UE 엔진 버전에 맞게 수정
	"Category": "",
	"Description": "",
	"Modules": [
		{
			"Name": "ENARobot",
			"Type": "Runtime",
			"LoadingPhase": "Default",
			"AdditionalDependencies": [ "Engine" ]
		}
	],
	"Plugins": [
        {
            "Name": "DigitalTwinCommPlugin", // 추가된 플러그인 이름
            "Enabled": true // 플러그인 활성화 필수
        }
		// ... 기타 플러그인 ...
	]
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

DefaultEngine.ini (ENARobot/Config/DefaultEngine.ini)

역할: 엔진 전반의 기본 설정. 네트워킹 관련 설정을 포함할 수 있지만, Python 통신 포트는 주로 컴포넌트 인스턴스에서 설정합니다.

코드: (UE 에디터 생성 기본 + 수정)

; ENARobot/Config/DefaultEngine.ini
[Engine]
; ... 기타 엔진 설정 ...

[Networking]
; ... 일반 네트워킹 설정 ...

; [/Script/DigitalTwinCommPlugin.PythonCommServerComponent] ; 컴포넌트 프로퍼티를 INI에서 관리하고 싶다면
; ListenPort=9999 ; (하지만 에디터에서 설정하는 경우가 흔함)
; ListenIP="127.0.0.1"
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Ini
IGNORE_WHEN_COPYING_END

DefaultGame.ini (ENARobot/Config/DefaultGame.ini)

역할: 게임플레이 전반의 기본 설정. 프로젝트 시작 시 로드될 기본 맵을 지정합니다.

코드: (UE 에디터 생성 기본 + 수정)

; ENARobot/Config/DefaultGame.ini
[/Script/EngineSettings.GameMapsSettings]
GameDefaultMap=/Game/Maps/YourLevel ; 로봇과 통신 매니저가 배치될 맵 경로
; EditorStartupMap=/Game/Maps/YourLevel
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Ini
IGNORE_WHEN_COPYING_END





2. C++ 소스 코드 파일

ENARobot.Build.cs (ENARobot/Source/ENARobot/ENARobot.Build.cs)

역할: 메인 게임 모듈 빌드 설정. 소켓 통신, JSON 처리, 커스텀 플러그인 모듈에 대한 의존성을 명시합니다.

코드: (UE 에디터 생성 기본 + 수정)

using UnrealBuildTool;
public class ENARobot : ModuleRules
{
    public ENARobot(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core", "CoreUObject", "Engine", "InputCore",
            "Sockets", // 추가: 소켓 통신
            "Networking", // 추가: 네트워킹
            "Json", // 추가: JSON 파싱
            "JsonUtilities", // 추가: JSON <-> UObject 변환 유틸리티
            "DigitalTwinCommPlugin" // 추가: 커스텀 플러그인 모듈
        });
        PrivateDependencyModuleNames.AddRange(new string[] { });
        // ... 기타 설정 ...
    }
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C#
IGNORE_WHEN_COPYING_END

ENARobot.h / ENARobot.cpp (ENARobot/Source/ENARobot/)

역할: 프로젝트의 메인 C++ 모듈 정의 및 구현. 모듈 로드/언로드 시 기본 코드 포함.

코드: (UE 에디터 생성 기본)

// ENARobot.h
#pragma once
#include "Modules/ModuleManager.h"
class FENARobotModule : public IModuleInterface { ... };
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END
// ENARobot.cpp
#include "ENARobot.h"
// ... 로깅 정의 ...
void FENARobotModule::StartupModule() { UE_LOG(LogTemp, Log, TEXT("FENARobotModule starting up.")); }
void FENARobotModule::ShutdownModule() { UE_LOG(LogTemp, Log, TEXT("FENARobotModule shutting down.")); }
IMPLEMENT_MODULE(FENARobotModule, ENARobot);
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END

RobotBaseActor.h / RobotBaseActor.cpp (ENARobot/Source/ENARobot/)

역할: 로봇 액터 C++ 기본 클래스. 로봇 ID, 메시/파티클 컴포넌트 참조, Python 명령 처리(SetRobotPoseFromJSON, HandleWeldingVisualCommand), 블루프린트에서 구현될 상세 제어 이벤트(ApplyJointAngles, SetArcVisibility 등) 선언/정의. BeginPlay 시 통신 컴포넌트에 자신을 등록하는 로직 포함.

코드: (UE 에디터 C++ 클래스 생성 + 수정 - 위에서 상세 코드 제공)

// RobotBaseActor.h
#pragma once
#include "GameFramework/Actor.h"
#include "Json.h" // FJsonObject
#include "Components/SkeletalMeshComponent.h"
#include "Particles/ParticleSystemComponent.h"
class UPythonCommServerComponent; // Forward Declaration
#include "RobotBaseActor.generated.h"
UCLASS() class ENAROBOT_API ARobotBaseActor : public AActor {
    GENERATED_BODY()
    UPROPERTY(...) int32 RobotID;
    UPROPERTY(...) USkeletalMeshComponent* RobotMesh;
    UPROPERTY(...) UParticleSystemComponent* WeldingArcParticle;
    UFUNCTION(BlueprintCallable) void SetRobotPoseFromJSON(const TSharedPtr<FJsonObject>& Params);
    UFUNCTION(BlueprintCallable) void HandleWeldingVisualCommand(const TSharedPtr<FJsonObject>& Params);
    UFUNCTION(BlueprintImplementableEvent) void ApplyJointAngles(const TArray<float>& Angles);
    UFUNCTION(BlueprintImplementableEvent) void SetArcVisibility(bool bVisible);
    // ... 기타 선언 ...
};
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END
// RobotBaseActor.cpp
#include "RobotBaseActor.h"
#include "PythonCommServerComponent.h" // Include actual header
#include "Kismet/GameplayStatics.h"
// ... 구현 ...
ARobotBaseActor::ARobotBaseActor() { /* 컴포넌트 생성/부착 */ }
void ARobotBaseActor::BeginPlay() { Super::BeginPlay(); /* PythonCommServerComponent 찾고 등록 */ }
void ARobotBaseActor::SetRobotPoseFromJSON(...) { /* JSON 파싱 및 ApplyJointAngles/ApplyTCPTransform 호출 */ }
void ARobotBaseActor::HandleWeldingVisualCommand(...) { /* JSON 파싱 및 SetArcVisibility/UpdateArcVisuals 호출 */ }
// ... 기타 함수 구현 ...
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END

PythonCommServerComponent.h / PythonCommServerComponent.cpp (ENARobot/Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/)

역할: Python 통신 서버 컴포넌트 C++ 클래스. TCP 서버 소켓 관리, 스레드 기반 수신, 메시지 프레이밍 파싱, JSON 파싱, 메시지 큐(게임 스레드 전달), 델리게이트 트리거, Python으로 응답 전송 로직 구현. FPythonMessage USTRUCT 정의.

코드: (UE 에디터 C++ 컴포넌트 생성 + 수정 - 위에서 상세 코드 제공)

// PythonCommServerComponent.h
#pragma once
#include "Components/ActorComponent.h"
#include "Networking.h" // Sockets, Networking
#include "Json.h" // Json parsing
#include "HAL/Runnable.h" // Background thread
#include "Containers/Queue.h" // Message queue
#include "PythonCommServerComponent.generated.h"
USTRUCT(BlueprintType) struct FPythonMessage { GENERATED_BODY() ... }; // Message structure
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(...) FOnPythonMessageReceived; // Delegate
UCLASS(...) class UPythonCommServerComponent : public UActorComponent {
    GENERATED_BODY()
    UPROPERTY(...) int32 ListenPort; // Editor-editable properties
    UFUNCTION(BlueprintCallable) bool StartServer();
    UFUNCTION(BlueprintCallable) void StopServer();
    UFUNCTION(BlueprintCallable) void RegisterRobotActor(int32 RobotId, AActor* RobotActor);
    // ... 기타 멤버 선언 ...
};
class FSocketReceiveWorker : public FRunnable { ... }; // Worker thread class
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END
// PythonCommServerComponent.cpp
#include "PythonCommServerComponent.h"
#include "Sockets.h"
#include "Networking.h"
#include "Json.h"
// ... 구현 ...
FSocketReceiveWorker::FSocketReceiveWorker(...) { /* Worker Init */ }
uint32 FSocketReceiveWorker::Run() { /* Receive loop, parsing, enqueueing */ return 0; }
void FSocketReceiveWorker::Stop() { /* Signal stop */ }
UPythonCommServerComponent::UPythonCommServerComponent() { /* Constructor */ }
void UPythonCommServerComponent::BeginPlay() { Super::BeginPlay(); StartServer(); }
void UPythonCommServerComponent::EndPlay(...) { Super::EndPlay(...); StopServer(); }
bool UPythonCommServerComponent::StartServer() { /* Create and start listener */ }
void UPythonCommServerComponent::StopServer() { /* Stop listener, worker, close socket */ }
bool UPythonCommServerComponent::OnSocketConnectionAccepted(...) { /* Handle new connection, start worker */ }
void UPythonCommServerComponent::TickComponent(...) { Super::TickComponent(...); ProcessReceivedMessages(); }
void UPythonCommServerComponent::ProcessReceivedMessages() { /* Dequeue messages, dispatch via OnMessageReceived */ }
bool UPythonCommServerComponent::SendFramedData(...) { /* Send data with length prefix */ }
bool UPythonCommServerComponent::SendJsonResponse(...) { /* Create JSON response, call SendFramedData */ }
void UPythonCommServerComponent::RegisterRobotActor(...) { /* Add actor to map */ }
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END

DigitalTwinCommPlugin.Build.cs (ENARobot/Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/)

역할: 플러그인 모듈 빌드 설정. 플러그인에 필요한 UE 모듈(Sockets, Networking, Json, JsonUtilities)에 의존성을 명시합니다.

코드: (플러그인 생성 시 자동 생성 기본 + 수정)

using UnrealBuildTool;
public class DigitalTwinCommPlugin : ModuleRules {
    public DigitalTwinCommPlugin(ReadOnlyTargetRules Target) : base(Target) {
        PublicDependencyModuleNames.AddRange(new string[] { "Core", "Sockets", "Networking", "Json", "JsonUtilities" });
        PrivateDependencyModuleNames.AddRange(new string[] { "CoreUObject", "Engine", "Slate", "SlateCore" });
        // ... 기타 설정 ...
    }
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C#
IGNORE_WHEN_COPYING_END

DigitalTwinCommPlugin.h / DigitalTwinCommPlugin.cpp (ENARobot/Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/)

역할: 플러그인 C++ 모듈 자체의 진입점. 모듈 로드/언로드 시 기본 코드.

코드: (플러그인 생성 시 자동 생성 기본)

// DigitalTwinCommPlugin.h
#pragma once
#include "Modules/ModuleManager.h"
class FDigitalTwinCommPluginModule : public IModuleInterface { ... };
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END
// DigitalTwinCommPlugin.cpp
#include "DigitalTwinCommPlugin.h"
// ... 로깅 정의 ...
void FDigitalTwinCommPluginModule::StartupModule() { UE_LOG(LogTemp, Log, TEXT("DigitalTwinCommPlugin starting up.")); }
void FDigitalTwinCommPluginModule::ShutdownModule() { UE_LOG(LogTemp, Log, TEXT("DigitalTwinCommPlugin shutting down.")); }
IMPLEMENT_MODULE(FDigitalTwinCommPluginModule, DigitalTwinCommPlugin);
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
C++
IGNORE_WHEN_COPYING_END









3. Blueprint 에셋 (에디터에서 생성 및 설정)

YourLevel.umap (ENARobot/Content/Maps/)

역할: 로봇 모델, 환경, 통신 매니저 액터 등이 배치될 3D 레벨 파일입니다. DefaultGame.ini의 GameDefaultMap으로 지정됩니다.

생성 방법: UE 에디터에서 "File" -> "New Level" 선택 후 원하는 템플릿(예: Empty)을 선택하여 생성하고 .umap 파일로 저장합니다.

구성:

BP_DigitalTwinManager 액터를 레벨에 배치합니다.

로봇 Blueprint 액터(BP_MyRobotActor 등)들을 레벨에 필요한 수량만큼 배치하고, 각 액터의 Robot ID 속성을 Python 시스템과 일치하도록 고유한 값(1, 2, 3, 4 등)으로 설정합니다.

환경 메시, 라이팅 등 시각화에 필요한 요소들을 구성합니다.

BP_DigitalTwinManager.uasset (ENARobot/Content/Blueprints/)

역할: UPythonCommServerComponent를 가지고 레벨에 배치되어 Python과의 통신 서버를 구동하고, 수신된 메시지를 실제 로봇 액터들에게 전달하는 역할을 합니다.

생성 방법: UE 에디터에서 "Content Browser" -> "Add" -> "Blueprint Class" 선택 후 "Actor"를 부모로 선택하여 생성하고 이름을 "BP_DigitalTwinManager"로 지정합니다. 블루프린트 에디터에서 이 액터에 UPythonCommServerComponent 컴포넌트를 추가합니다.

구성:

UPythonCommServerComponent 컴포넌트를 선택하고 Details 패널에서 Listen IP 및 Listen Port 속성을 Python config.py의 UE_SIMULATOR_IP, UE_SIMULATOR_PORT 값과 동일하게 설정합니다.

Event Graph:

"Event BeginPlay" 노드에서 UPythonCommServerComponent의 StartServer() 함수를 호출합니다.

UPythonCommServerComponent 컴포넌트의 "On Client Connected", "On Client Disconnected" 이벤트에 바인딩하여 연결/해제 시 로깅 또는 다른 처리를 수행합니다.

UPythonCommServerComponent 컴포넌트의 "On Message Received" 이벤트에 바인딩합니다. 이 이벤트가 발생하면 전달된 FPythonMessage 구조체에서 Action (문자열), RobotId (정수), Parameters (JSON Object)를 가져옵니다. RobotId를 사용하여 레벨에 배치된 모든 로봇 액터(BP_MyRobotActor 등) 중에서 해당 ID를 가진 액터를 찾고, Action 필드에 따라 해당 로봇 액터의 적절한 함수(예: SetRobotPoseFromJSON, HandleWeldingVisualCommand)를 호출하고 Parameters JSON 오브젝트를 전달합니다. (레벨의 모든 액터를 순회하거나, PythonCommServerComponent의 RegisteredRobots 맵을 활용하는 등의 방법으로 로봇 액터를 찾습니다).

BP_MyRobotActor.uasset (ENARobot/Content/Blueprints/)

역할: ARobotBaseActor C++ 기본 클래스를 상속받아 실제 로봇 3D 모델 에셋을 할당하고, C++에서 선언된 BlueprintImplementableEvent 함수들을 오버라이드하여 로봇 모델의 상세 제어 및 시각 효과 로직을 구현합니다.

생성 방법: UE 에디터에서 "Content Browser" -> "Add" -> "Blueprint Class" 선택 후 "All Classes"에서 ARobotBaseActor (또는 상속받은 다른 C++ 로봇 클래스)를 검색하여 부모로 선택하고 이름을 "BP_MyRobotActor"로 지정합니다.

구성:

디테일 패널에서 Robot ID 속성을 이 액터 인스턴스의 고유 ID로 설정합니다 (예: 로봇 1번이면 1, 로봇 2번이면 2).

RobotMesh 컴포넌트에 실제 로봇 스켈레탈 메시 에셋을 할당합니다.

WeldingArcParticle 컴포넌트에 용접 아크 파티클 시스템 에셋을 할당하고, 로봇 메시의 "WeldingTipSocket" (또는 다른 지정된 소켓)에 부착합니다.

Event Graph:

"Event BeginPlay" 노드에서 부모(ARobotBaseActor)의 BeginPlay를 호출하도록 합니다 (Parent: BeginPlay). 이렇게 해야 C++ ARobotBaseActor::BeginPlay에서 RegisterRobotActor가 호출됩니다.

오버라이드된 BlueprintImplementableEvent 함수 구현:

ApplyJointAngles: 입력받은 Joint Angles 배열(float의 TArray)을 사용하여 RobotMesh 컴포넌트의 각 스켈레탈 본(Bone)의 상대 회전(Relative Rotation)을 설정합니다. 각 관절의 본 이름(FName)과 회전 축(Roll, Pitch, Yaw)에 맞춰 각도 값을 적용하는 로직이 필요합니다. 애니메이션 블루프린트와 연동하여 제어 본(Control Bone)의 트랜스폼을 업데이트하는 방식이 권장됩니다.

ApplyTCPTransform: 입력받은 Location (FVector)과 Rotation (FRotator 또는 FQuat)을 사용하여 RobotMesh 컴포넌트 (또는 로봇의 베이스 본)의 월드 트랜스폼을 설정하거나, 복잡한 Inverse Kinematics (IK) 솔버를 사용하여 로봇이 해당 TCP 위치/방향을 따르도록 제어하는 로직을 구현합니다. 좌표계 및 단위 변환에 유의해야 합니다.

SetArcVisibility: 입력받은 boolean 값에 따라 WeldingArcParticle 컴포넌트의 SetVisibility 함수를 호출합니다.

UpdateArcVisuals: 입력받은 Visual Details (FJsonObject)에서 Python이 보낸 용접 파라미터(전류, 전압 등)를 추출하고, 이 값에 따라 WeldingArcParticle 컴포넌트의 파티클 파라미터(예: 색상, 크기, 스폰 속도)를 동적으로 변경하는 로직을 구현합니다. (JSON Object에서 특정 필드 값을 추출하는 Blueprint 함수 사용)

기타 에셋: 로봇의 3D 모델(스켈레탈 메시), 용접 팁 메시, 용접 아크 파티클 시스템, 용접 비드 모델 등 시각화에 사용될 모든 3D 에셋들이 Content/ 디렉토리 내 관련 폴더에 임포트 또는 생성되어 BP_MyRobotActor 블루프린트에서 사용되도록 설정되어야 합니다.
