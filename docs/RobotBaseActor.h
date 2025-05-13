// Header file (RobotBaseActor.h)

#pragma once

#include "GameFramework/Actor.h"
#include "Json.h" // For JSON parsing if you parse parameters here
#include "Components/SkeletalMeshComponent.h" // For accessing bones
#include "Particles/ParticleSystemComponent.h" // For welding arc
#include "RobotBaseActor.generated.h"

UCLASS()
class YOURUNREALMODULE_API ARobotBaseActor : public AActor
{
    GENERATED_BODY()

public:
    ARobotBaseActor();

    // Public variable to set Robot ID in the editor/Blueprint
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Robot")
    int32 RobotID = -1; // Unique ID matching the Python system's robot_id

    // Reference to the Skeletal Mesh Component
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    USkeletalMeshComponent* RobotMesh;

    // Reference to the Welding Arc Particle Component
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    UParticleSystemComponent* WeldingArcParticle;

    // Optional: Reference to a Static Mesh Component representing the welding tip
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Components")
    UStaticMeshComponent* WeldingTipMesh;


    virtual void BeginPlay() override;

    // Function to set robot pose from JSON parameters (called from PythonCommServerComponent)
    UFUNCTION(BlueprintCallable, Category = "Robot Control")
    void SetRobotPoseFromJSON(const TSharedPtr<FJsonObject>& PoseParameters);

    // Function to handle welding visual commands from JSON parameters
    UFUNCTION(BlueprintCallable, Category = "Robot Control")
    void HandleWeldingVisualCommand(const TSharedPtr<FJsonObject>& CommandParameters);


protected:
    // Internal function to apply joint angles
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void ApplyJointAngles(const TArray<float>& JointAngles);

    // Internal function to apply TCP transform
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void ApplyTCPTransform(const FVector& Location, const FQuat& Rotation); // Or FRotator

    // Internal function to set arc visibility
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void SetArcVisibility(bool bVisible);

    // Internal function to update arc parameters (color, size, etc.)
    UFUNCTION(BlueprintImplementableEvent, Category = "Robot Internal")
    void UpdateArcVisuals(const TSharedPtr<FJsonObject>& VisualDetails);


private:
    // Add references to bones if needed (e.g., using FName)
    // FName JointBoneNames[6]; // Example for a 6-DOF robot

};