// Source/ENARobot/RobotBaseActor.h

#pragma once

#include "GameFramework/Actor.h"
#include "Json.h" // For FJsonObject
#include "Components/SkeletalMeshComponent.h" // For accessing bones
#include "Particles/ParticleSystemComponent.h" // For welding arc

// 순환 참조 방지를 위해 헤더 대신 클래스 전방 선언 (Forward Declaration) 사용
class UPythonCommServerComponent; // PhysicsInterface 역할을 할 컴포넌트

#include "RobotBaseActor.generated.h" // UCLASS, UPROPERTY, UFUNCTION 매크로 처리

UCLASS()
class ENAROBOT_API ARobotBaseActor : public AActor
{
    GENERATED_BODY()

public:
    ARobotBaseActor(); // 생성자

    // 액터의 각 프레임 업데이트 (필요시 활성화)
    // virtual void Tick(float DeltaTime) override;

    // 게임 시작 시 호출
    virtual void BeginPlay() override;

    // --- 속성 (에디터에서 편집 가능, 블루프린트에서 접근 가능) ---
    // Python 시스템의 Robot ID와 매핑될 고유 ID
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Robot")
    int32 RobotID = -1;

    // 로봇 3D 모델을 위한 스켈레탈 메시 컴포넌트 참조
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    USkeletalMeshComponent* RobotMesh;

    // 용접 아크 시각화를 위한 파티클 시스템 컴포넌트 참조
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    UParticleSystemComponent* WeldingArcParticle;

    // (선택적) 용접 팁 메시 컴포넌트 참조 (파티클 부착 위치 지정 등에 사용)
    // UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    // UStaticMeshComponent* WeldingTipMesh;


    // --- 함수 (PythonCommServerComponent에서 호출될) ---
    // Python에서 받은 'set_robot_pose' 명령을 처리하는 함수
    // BlueprintCallable: 블루프린트에서 이 함수를 호출 가능
    UFUNCTION(BlueprintCallable, Category = "Robot Control")
    void SetRobotPoseFromJSON(const TSharedPtr<FJsonObject>& PoseParameters);

    // Python에서 받은 'welding_visual_command' 명령을 처리하는 함수
    UFUNCTION(BlueprintCallable, Category = "Robot Control")
    void HandleWeldingVisualCommand(const TSharedPtr<FJsonObject>& CommandParameters);


protected:
    // --- Blueprint Implementable Events (블루프린트에서 구현될 상세 로직) ---
    // 관절 각도 적용 로직 (블루프린트에서 구현)
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void ApplyJointAngles(const TArray<float>& JointAngles);

    // TCP 트랜스폼 (위치/회전) 적용 로직 (블루프린트에서 구현)
    // FRotator 또는 FQuat 중 Python에서 보내는 형식과 일치시키거나 변환 로직 필요
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void ApplyTCPTransform(const FVector& Location, const FRotator& Rotation); // 예시: FRotator 사용

    // 용접 아크 파티클 가시성 설정 로직 (블루프린트에서 구현)
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void SetArcVisibility(bool bVisible);

    // 용접 아크 파티클 파라미터 업데이트 로직 (블루프린트에서 구현)
    // JSON 파라미터는 전류, 전압 등 파티클 효과에 영향을 줄 데이터를 담을 수 있음
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void UpdateArcVisuals(const TSharedPtr<FJsonObject>& VisualDetails);


private:
    // BegginPlay 시 레벨에서 PythonCommServerComponent를 찾아 참조를 저장할 수도 있습니다.
    // UPROPERTY()
    // UPythonCommServerComponent* PythonCommServer; // 참조 저장 (GC 대상이 아님)
};