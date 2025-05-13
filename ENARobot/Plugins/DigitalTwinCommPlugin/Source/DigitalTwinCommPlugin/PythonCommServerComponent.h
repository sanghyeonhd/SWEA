// Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/PythonCommServerComponent.h

#pragma once

#include "Components/ActorComponent.h"
#include "Networking.h" // For FSocket, FTcpSocketListener etc.
#include "Json.h" // For JSON parsing
#include "HAL/Runnable.h" // For background thread
#include "Containers/Queue.h" // For message queue
#include "HAL/RunnableThread.h" // For FRunnableThread
#include "Misc/Base64.h" // For Base64 encoding/decoding (if sending binary data like images)
#include "Serialization/JsonReader.h" // For TJsonReader
#include "Serialization/JsonSerializer.h" // For FJsonSerializer
#include "Kismet/BlueprintFunctionLibrary.h" // For BlueprintCallable static functions if needed
#include "JsonUtilities.h" // For FJsonObjectConverter (useful for converting JSON to UStruct/UObject)


#include "PythonCommServerComponent.generated.h" // UCLASS, UPROPERTY, UFUNCTION 매크로 처리

// Python 메시지 구조체 정의 (Blueprint에서 접근 가능하도록 USTRUCT로 만듦)
USTRUCT(BlueprintType)
struct FPythonMessage
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "Python Comm")
    FString Action; // e.g., "set_robot_pose", "welding_visual_command", "run_simulation"

    UPROPERTY(BlueprintReadOnly, Category = "Python Comm")
    int32 RobotId; // Associated Robot ID (-1 if not applicable)

    // JSON Parameters를 블루프린트로 바로 노출하기는 어려움.
    // 필요시 특정 파라미터들을 별도 UPROPERTY로 추출하거나,
    // 블루프린트에서 FJsonObject를 처리하는 유틸리티 함수를 사용해야 함.
    // TSharedPtr<FJsonObject> Parameters; // UPROPERTY로 노출 불가

    // Sequence ID (응답 추적 필요시)
    // UPROPERTY(BlueprintReadOnly, Category = "Python Comm")
    // int32 SequenceId = -1;

    // Raw JSON string (옵션)
    // FString RawJsonString; // UPROPERTY로 노출 불가
};


// 메시지 수신 시 호출될 델리게이트 정의 (Blueprint에서 바인딩 가능)
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPythonMessageReceived, FPythonMessage, Message); // USTRUCT 사용

// 연결 이벤트 델리게이트 정의
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnPythonClientConnected);
DECLARE_DYNAMIC_MULTICAST_DEL<ctrl63>