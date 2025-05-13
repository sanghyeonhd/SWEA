// Header file (PythonCommServerComponent.h)

#pragma once

#include "Components/ActorComponent.h"
#include "Networking.h" // For FSocket, FTcpSocketListener etc.
#include "Json.h" // For JSON parsing
#include "HAL/Runnable.h" // For background thread
#include "Containers/Queue.h" // For message queue

#include "PythonCommServerComponent.generated.h"

// Define a struct to hold parsed messages for processing on the game thread
struct FPythonMessage
{
    FString Action;
    int32 RobotId; // Or FString RobotName
    TSharedPtr<FJsonObject> Parameters;
    // Add Sequence ID if you need to track responses
};

// Declare a delegate to pass received messages from the worker thread to the game thread
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPythonMessageReceived, const FPythonMessage&, Message);

// Declare a delegate for connection events
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnPythonClientConnected);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnPythonClientDisconnected);


UCLASS( ClassGroup=(DigitalTwin), meta=(BlueprintSpawnableComponent) )
class YOURUNREALMODULE_API UPythonCommServerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPythonCommServerComponent();
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override; // Used to process messages from queue

    // --- Configuration ---
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Python Comm")
    FString ListenIP = TEXT("127.0.0.1"); // IP address to listen on

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Python Comm")
    int32 ListenPort = 9999; // Port to listen on (must match physics_interface.py)

    // --- Events (exposed to Blueprint) ---
    UPROPERTY(BlueprintAssignable, Category = "Python Comm")
    FOnPythonMessageReceived OnMessageReceived; // Event triggered when a message is received and parsed

    UPROPERTY(BlueprintAssignable, Category = "Python Comm")
    FOnPythonClientConnected OnClientConnected;

    UPROPERTY(BlueprintAssignable, Category = "Python Comm")
    FOnPythonClientDisconnected OnClientDisconnected;


    // --- Public Interface (Blueprint Callable) ---
    UFUNCTION(BlueprintCallable, Category = "Python Comm")
    bool StartServer();

    UFUNCTION(BlueprintCallable, Category = "Python Comm")
    void StopServer();

    // --- Utility for Blueprint to find registered robots ---
    // Need a way to register robots with this component
    // Example: BP_MyRobot calls this on BeginPlay
    UFUNCTION(BlueprintCallable, Category = "Python Comm")
    void RegisterRobotActor(int32 RobotId, AActor* RobotActor);

    // --- Send Response Back to Python (if needed for specific actions) ---
    UFUNCTION(BlueprintCallable, Category = "Python Comm")
    bool SendJsonResponse(int32 ClientConnectionId, const FString& Status, const FString& Message, const TSharedPtr<FJsonObject>& Data);


private:
    FTcpSocketListener* TcpListener;
    FSocket* ClientSocket; // We'll assume one client connection for simplicity first
    // For multiple clients, you would need TArray<FSocket*> ClientSockets and a handler per socket

    // Worker thread for receiving data
    class FSocketReceiveWorker : public FRunnable
    {
    public:
        FSocketReceiveWorker(FSocket* InSocket, TQueue<FPythonMessage>& InMessageQueue, FEvent* InStopEvent);
        virtual bool Init() override;
        virtual uint32 Run() override;
        virtual void Stop() override;
        virtual void Exit() override;

    private:
        FSocket* ClientSocket;
        TQueue<FPythonMessage>& MessageQueue; // Queue to send messages to the game thread
        FEvent* StopEvent; // Event to signal the thread to stop
        FThreadSafeBool bIsRunning; // Flag to check if the thread should be running
    };
    FSocketReceiveWorker* ReceiveWorker;
    FRunnableThread* ReceiveThread;
    FEvent* ReceiveStopEvent; // Event object for the worker thread

    TQueue<FPythonMessage> ReceivedMessageQueue; // Queue to pass parsed messages to game thread

    // Map to store registered robot actors
    TMap<int32, AActor*> RegisteredRobots; // Mapping Robot ID (from Python) to UE Actor

    // Internal function to handle new client connections
    bool OnSocketConnectionAccepted(FSocket* ClientSocket, const FIPv4Endpoint& ClientEndpoint);

    // Internal function to process messages from the queue on the game thread
    void ProcessReceivedMessages();

    // Internal helper to send framed data
    bool SendFramedData(const FString& DataToSend);
};