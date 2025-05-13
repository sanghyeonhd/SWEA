// Source/ENARobot/RobotBaseActor.cpp

#include "RobotBaseActor.h"
#include "PythonCommServerComponent.h" // 통신 컴포넌트 헤더 포함
#include "Kismet/GameplayStatics.h" // UGameplayStatics 사용
#include "JsonUtilities.h" // FJsonObjectConverter 사용 (필요시)

// 생성자: 컴포넌트 생성 및 기본 설정
ARobotBaseActor::ARobotBaseActor()
{
    // 각 프레임 업데이트 필요시 활성화 (BP에서 Tick 이벤트를 사용하면 C++ Tick은 꺼도 됨)
    PrimaryActorTick.bCanEverTick = true;

    // 스켈레탈 메시 컴포넌트 생성 및 RootComponent로 설정
    RobotMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("RobotSkeletalMesh"));
    RootComponent = RobotMesh;

    // 용접 아크 파티클 시스템 컴포넌트 생성 및 메시의 소켓에 부착
    WeldingArcParticle = CreateDefaultSubobject<UParticleSystemComponent>(TEXT("WeldingArcParticle"));
    // "WeldingTipSocket"은 로봇 스켈레탈 메시의 용접 팁 위치에 정의된 소켓 이름이어야 합니다.
    WeldingArcParticle->SetupAttachment(RobotMesh, TEXT("WeldingTipSocket"));
    WeldingArcParticle->SetVisibility(false); // 초기에는 보이지 않게 설정
    WeldingArcParticle->bAutoActivate = false; // 자동으로 활성화되지 않게 설정

    // (선택적) 용접 팁 메시 컴포넌트
    // WeldingTipMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WeldingTipMesh"));
    // WeldingTipMesh->SetupAttachment(RobotMesh, TEXT("WeldingTipSocket"));

    // 로봇 스켈레탈 메시 에셋은 이 클래스를 상속받는 블루프린트에서 설정해야 합니다.
    // 예시: RobotMesh->SetSkeletalMesh(LoadMeshFromPath("..."));
}

// 게임 시작 시 호출
void ARobotBaseActor::BeginPlay()
{
    Super::BeginPlay();

    // PythonCommServerComponent를 찾아 자신을 등록합니다.
    // 레벨에 배치된 모든 Actor를 검사하여 해당 컴포넌트를 가진 액터를 찾습니다.
    UPythonCommServerComponent* CommComponent = nullptr;
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AActor::StaticClass(), FoundActors); // 모든 액터 검색 (비효율적일 수 있음)
    // 더 나은 방법: 컴포넌트 자체에 static 함수를 만들어 싱글톤처럼 접근하거나, GameState/GameMode에 참조를 저장
    // 예시에서는 간단하게 모든 액터 검색 후 컴포넌트 찾기
    for (AActor* Actor : FoundActors)
    {
        CommComponent = Actor->FindComponentByClass<UPythonCommServerComponent>();
        if (CommComponent)
        {
            // 컴포넌트를 찾았으면 이 로봇 액터를 등록합니다.
            CommComponent->RegisterRobotActor(RobotID, this);
            // PythonCommServer = CommComponent; // 참조 저장 (선택적)
            break; // 컴포넌트는 하나만 있다고 가정
        }
    }
    if (!CommComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("ARobotBaseActor '%s' (ID %d): UPythonCommServerComponent not found in level! Cannot register."), *GetName(), RobotID);
    }
}

// Python에서 받은 'set_robot_pose' 명령을 처리하는 함수 구현
void ARobotBaseActor::SetRobotPoseFromJSON(const TSharedPtr<FJsonObject>& PoseParameters)
{
    if (!PoseParameters.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("SetRobotPoseFromJSON called with invalid parameters for Robot %d."), RobotID);
        return;
    }

    // --- JSON 파라미터에서 포즈 데이터 추출 및 적용 ---
    // Python에서 보낸 JSON 구조와 일치해야 합니다: {"joint_angles": [...]} 또는 {"tcp_transform": {"position": [...], "rotation": [...]}}

    const TArray<TSharedPtr<FJsonValue>>* JointAnglesArray;
    if (PoseParameters->TryGetArrayField(TEXT("joint_angles"), JointAnglesArray))
    {
        // 관절 각도 적용 (블루프린트 이벤트 호출)
        TArray<float> JointAngles;
        if (JointAnglesArray)
        {
            for (const TSharedPtr<FJsonValue>& JsonValue : *JointAnglesArray)
            {
                double Angle;
                if (JsonValue->TryGetNumber(Angle))
                {
                    JointAngles.Add(static_cast<float>(Angle));
                }
                // 유효하지 않은 값에 대한 처리 로직 추가 필요
            }
        }

        // 블루프린트 이벤트 호출 또는 C++ 로직으로 직접 관절 회전 설정
        // 관절 개수 확인 등 유효성 검사 추가 필요
        if (JointAngles.Num() > 0)
        {
             ApplyJointAngles(JointAngles); // <-- 블루프린트 이벤트 호출
             UE_LOG(LogTemp, Verbose, TEXT("Robot %d: Applied %d joint angles."), RobotID, JointAngles.Num());
        }
         else
         {
             UE_LOG(LogTemp, Warning, TEXT("Robot %d: Received empty or invalid joint angles array."), RobotID);
         }
    }

    const TSharedPtr<FJsonObject>* TcpTransformObject;
    if (PoseParameters->TryGetObjectField(TEXT("tcp_transform"), TcpTransformObject) && TcpTransformObject && TcpTransformObject->IsValid())
    {
        // TCP 트랜스폼 적용 (위치 및 회전)
        const TArray<TSharedPtr<FJsonValue>>* PositionArray;
        const TArray<TSharedPtr<FJsonValue>>* RotationArray; // 형식은 Python과 일치해야 함

        FVector Location = FVector::ZeroVector;
        FRotator Rotation = FRotator::ZeroRotator; // 예시: FRotator 사용. Quaternion 사용시 FQuat.

        // 위치 추출 (예시: [x, y, z] 배열)
        if ((*TcpTransformObject)->TryGetArrayField(TEXT("position"), PositionArray) && PositionArray && PositionArray->Num() >= 3)
        {
            // UE 좌표계와 Python/로봇 좌표계 간의 변환이 필요할 수 있습니다.
            // 예시: Python의 X, Y, Z가 UE의 X, Y, Z와 다를 경우
            Location.X = static_cast<float>((*PositionArray)[0]->AsNumber());
            Location.Y = static_cast<float>((*PositionArray)[1]->AsNumber());
            Location.Z = static_cast<float>((*PositionArray)[2]->AsNumber());
            // TODO: 좌표계 변환 로직 구현
        } else { UE_LOG(LogTemp, Warning, TEXT("Robot %d: Missing or invalid 'position' array for TCP transform."), RobotID); }

        // 회전 추출 (예시: Quaternion [x, y, z, w] 배열)
        if ((*TcpTransformObject)->TryGetArrayField(TEXT("rotation"), RotationArray) && RotationArray && RotationArray->Num() >= 4)
        {
            FQuat QuatRotation;
             QuatRotation.X = static_cast<float>((*RotationArray)[0]->AsNumber());
             QuatRotation.Y = static_cast<float>((*RotationArray)[1]->AsNumber());
             QuatRotation.Z = static_cast<float>((*RotationArray)[2]->AsNumber());
             QuatRotation.W = static_cast<float>((*RotationArray)[3]->AsNumber());
             QuatRotation.Normalize(); // 쿼터니언 정규화

             Rotation = QuatRotation.Rotator(); // 쿼터니언을 FRotator로 변환 (또는 FRotator로 직접 받거나 FQuat 그대로 사용)
             // TODO: 회전 형식(Euler, Quaternion) 및 축 변환 로직 구현
        } // Else: 회전 데이터 없음, 기본값 사용

        // 블루프린트 이벤트 호출 또는 C++ 로직으로 직접 액터/컴포넌트 트랜스폼 설정
        ApplyTCPTransform(Location, Rotation); // <-- 블루프린트 이벤트 호출
        UE_LOG(LogTemp, Verbose, TEXT("Robot %d: Applied TCP transform. Location: %s, Rotation: %s"), RobotID, *Location.ToString(), *Rotation.ToString());
    }

    // joint_angles와 tcp_transform 둘 다 없으면 경고
     if (!JointAnglesArray && (!TcpTransformObject || !TcpTransformObject.IsValid()))
     {
          UE_LOG(LogTemp, Warning, TEXT("Robot %d: Received 'set_robot_pose' message but no valid 'joint_angles' or 'tcp_transform' parameters found."), RobotID);
     }
}


// Python에서 받은 'welding_visual_command' 명령을 처리하는 함수 구현
void ARobotBaseActor::HandleWeldingVisualCommand(const TSharedPtr<FJsonObject>& CommandParameters)
{
    if (!CommandParameters.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("HandleWeldingVisualCommand called with invalid parameters for Robot %d."), RobotID);
        return;
    }

    FString CommandType;
    if (!CommandParameters->TryGetStringField(TEXT("command_type"), CommandType))
    {
        UE_LOG(LogTemp, Warning, TEXT("Robot %d: Received welding_visual_command missing 'command_type' field."), RobotID);
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Robot %d received welding visual command: %s"), RobotID, *CommandType);

    // --- 명령 타입에 따른 처리 ---
    if (CommandType == TEXT("arc_on"))
    {
        SetArcVisibility(true); // <-- 블루프린트 이벤트 호출
         // Optional: Update arc parameters if details are provided
        TSharedPtr<FJsonObject> Details = CommandParameters->GetObjectField(TEXT("details"));
        if (Details.IsValid()) {
             UpdateArcVisuals(Details); // <-- 블루프린트 이벤트 호출
        }
    }
    else if (CommandType == TEXT("arc_off"))
    {
        SetArcVisibility(false); // <-- 블루프린트 이벤트 호출
    }
    // TODO: 다른 시각 명령 추가 (예: "set_bead_visibility", "update_bead_shape")
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Robot %d: Unknown welding visual command type: %s."), RobotID, *CommandType);
    }
}

// Blueprint Implementable Event 함수들은 블루프린트 에디터에서 오버라이드하여 구현해야 합니다.
// 예: ApplyJointAngles 구현 -> 블루프린트에서 SkeletalMeshComponent의 SetBoneRotation 또는 SetBoneTransform 사용