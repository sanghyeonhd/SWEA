// Source file (RobotBaseActor.cpp) - Key implementations

#include "RobotBaseActor.h"
#include "JsonUtilities.h" // For FJsonObjectConverter (optional, for complex object conversion)
#include "Animation/SkeletalMeshActor.h" // If inheriting from SkeletalMeshActor
#include "Animation/AnimInstance.h" // If using Animation Blueprint

ARobotBaseActor::ARobotBaseActor()
{
    PrimaryActorTick.bCanEverTick = true; // Allow ticking if needed

    // Create and attach components (example - customize based on your setup)
    RobotMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("RobotSkeletalMesh"));
    RootComponent = RobotMesh;

    WeldingArcParticle = CreateDefaultSubobject<UParticleSystemComponent>(TEXT("WeldingArcParticle"));
    // Attach particle system to a socket on the mesh (e.g., "WeldingTipSocket")
    WeldingArcParticle->SetupAttachment(RobotMesh, TEXT("WeldingTipSocket"));
    WeldingArcParticle->SetVisibility(false); // Start hidden
    WeldingArcParticle->bAutoActivate = false; // Don't activate automatically

    // Optional: Welding tip mesh
    // WeldingTipMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WeldingTipMesh"));
    // WeldingTipMesh->SetupAttachment(RobotMesh, TEXT("WeldingTipSocket")); // Attach to the same socket

    // Initialize bone names if using direct bone control
    // JointBoneNames[0] = TEXT("Joint1_Bone"); // Example names - MUST match your skeletal mesh

    // Ensure the skeletal mesh is set up in the Blueprint inheriting from this class
}

void ARobotBaseActor::BeginPlay()
{
    Super::BeginPlay();

    // Register this robot actor with the communication server component
    // Find the manager actor or component in the level
    UPythonCommServerComponent* CommComponent = nullptr;
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AActor::StaticClass(), FoundActors); // Find all actors
    for (AActor* Actor : FoundActors)
    {
        CommComponent = Actor->FindComponentByClass<UPythonCommServerComponent>();
        if (CommComponent)
        {
            // Found the component, register this robot
            CommComponent->RegisterRobotActor(RobotID, this);
            break;
        }
    }
    if (!CommComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("ARobotBaseActor '%s': PythonCommServerComponent not found in level! Cannot register."), *GetName());
    }
}

void ARobotBaseActor::SetRobotPoseFromJSON(const TSharedPtr<FJsonObject>& PoseParameters)
{
    if (!PoseParameters.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("SetRobotPoseFromJSON called with invalid parameters for Robot %d."), RobotID);
        return;
    }

    // --- Extract and apply pose data based on the JSON structure ---
    // Python sends: {"joint_angles": [...]} or {"tcp_transform": {"position": [...], "rotation": [...]}}

    const TArray<TSharedPtr<FJsonValue>>* JointAnglesArray;
    if (PoseParameters->TryGetArrayField(TEXT("joint_angles"), JointAnglesArray))
    {
        // Handle Joint Angles
        TArray<float> JointAngles;
        for (const TSharedPtr<FJsonValue>& JsonValue : *JointAnglesArray)
        {
            double Angle;
            if (JsonValue->TryGetNumber(Angle))
            {
                JointAngles.Add(static_cast<float>(Angle));
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Invalid joint angle value in array for Robot %d."), RobotID);
                // Decide how to handle invalid data - skip this update? Log error and continue?
                break; // Stop processing this message if invalid data found
            }
        }
        if (JointAngles.Num() > 0 && JointAngles.Num() == 6) // Basic validation: check if 6 angles received
        {
            // Call the Blueprint Implementable Event or C++ function to apply joint angles
            ApplyJointAngles(JointAngles); // <-- This calls the Blueprint function (if implemented)
             UE_LOG(LogTemp, Verbose, TEXT("Applied joint angles for Robot %d."), RobotID);
        }
         else if (JointAngles.Num() > 0)
         {
              UE_LOG(LogTemp, Warning, TEXT("Received %d joint angles for Robot %d, expected 6."), JointAngles.Num(), RobotID);
         }
    }

    const TSharedPtr<FJsonObject>* TcpTransformObject;
    if (PoseParameters->TryGetObjectField(TEXT("tcp_transform"), TcpTransformObject) && TcpTransformObject && TcpTransformObject->IsValid())
    {
        // Handle TCP Transform (Position and Rotation)
        const TArray<TSharedPtr<FJsonValue>>* PositionArray;
        const TArray<TSharedPtr<FJsonValue>>* RotationArray; // Format (Quaternion or Euler?) must match Python

        FVector Location = FVector::ZeroVector;
        FQuat Rotation = FQuat::Identity; // Or FRotator

        // Get Position (assumed [x, y, z])
        if ((*TcpTransformObject)->TryGetArrayField(TEXT("position"), PositionArray) && PositionArray && PositionArray->Num() == 3)
        {
             // Need to convert coordinate systems if Python's Z is UE's Z etc.
             Location.X = static_cast<float>((*PositionArray)[0]->AsNumber()); // Example mapping X
             Location.Y = static_cast<float>((*PositionArray)[1]->AsNumber()); // Example mapping Y
             Location.Z = static_cast<float>((*PositionArray)[2]->AsNumber()); // Example mapping Z
             // TODO: Implement coordinate system and unit conversion if needed!
        } else { UE_LOG(LogTemp, Warning, TEXT("Missing or invalid 'position' array for TCP transform for Robot %d."), RobotID); }


        // Get Rotation (assumed Quaternion [x, y, z, w]) - MUST match Python format!
        if ((*TcpTransformObject)->TryGetArrayField(TEXT("rotation"), RotationArray) && RotationArray && RotationArray->Num() == 4)
        {
             // Need to convert coordinate systems and rotation order if needed!
             Rotation.X = static_cast<float>((*RotationArray)[0]->AsNumber());
             Rotation.Y = static_cast<float>((*RotationArray)[1]->AsNumber());
             Rotation.Z = static_cast<float>((*RotationArray)[2]->AsNumber());
             Rotation.W = static_cast<float>((*RotationArray)[3]->AsNumber());
             Rotation.Normalize(); // Normalize Quaternion
             // TODO: Implement coordinate system and rotation conversion if needed!
        } // Else: Missing or invalid rotation array, use default identity rotation


        // Call the Blueprint Implementable Event or C++ function to apply TCP transform
        ApplyTCPTransform(Location, Rotation); // <-- This calls the Blueprint function (if implemented)
        UE_LOG(LogTemp, Verbose, TEXT("Applied TCP transform for Robot %d. Location: %s"), RobotID, *Location.ToString());

    }

    // If neither joint_angles nor tcp_transform was successfully parsed, log a warning.
    if (!JointAnglesArray && !TcpTransformObject)
    {
         UE_LOG(LogTemp, Warning, TEXT("Received 'set_robot_pose' message but no valid 'joint_angles' or 'tcp_transform' found for Robot %d."), RobotID);
    }
}


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
        UE_LOG(LogTemp, Warning, TEXT("Received welding_visual_command missing 'command_type' field for Robot %d."), RobotID);
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Robot %d received welding visual command: %s"), RobotID, *CommandType);

    // --- Handle Specific Command Types ---
    if (CommandType == TEXT("arc_on"))
    {
        SetArcVisibility(true); // Call Blueprint Implementable Event or C++ function
         // Optional: Update arc parameters if details are provided
        TSharedPtr<FJsonObject> Details = CommandParameters->GetObjectField(TEXT("details"));
        if (Details.IsValid()) {
             UpdateArcVisuals(Details); // e.g., update particle size/color based on current/voltage
        }
    }
    else if (CommandType == TEXT("arc_off"))
    {
        SetArcVisibility(false); // Call Blueprint Implementable Event or C++ function
    }
    // Add other visual commands as needed (e.g., "set_bead_visibility", "update_bead_shape")
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Unknown welding visual command type: %s for Robot %d."), *CommandType, RobotID);
    }
}

// Implement Blueprint Implementable Events in Blueprint Editor:
// Right-click on the Blueprint class -> Override Function -> Find "ApplyJointAngles", "ApplyTCPTransform", etc.