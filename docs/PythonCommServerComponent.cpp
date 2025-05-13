// Source file (PythonCommServerComponent.cpp) - Key implementations

#include "PythonCommServerComponent.h"
#include "Kismet/GameplayStatics.h" // For finding actors
#include "TimerManager.h" // For TickComponent timer (alternative to TickComponent)
#include "HAL/RunnableThread.h"
#include "Misc/Base64.h" // If sending/receiving binary data like images

// Include your Robot Actor base class header if you have one
// #include "YourRobotBaseActor.h"

// --- FSocketReceiveWorker Implementation ---
FSocketReceiveWorker::FSocketReceiveWorker(FSocket* InSocket, TQueue<FPythonMessage>& InMessageQueue, FEvent* InStopEvent)
    : ClientSocket(InSocket), MessageQueue(InMessageQueue), StopEvent(InStopEvent)
{
    bIsRunning = true;
}

bool FSocketReceiveWorker::Init()
{
    UE_LOG(LogTemp, Log, TEXT("FSocketReceiveWorker Init"));
    return true;
}

uint32 FSocketReceiveWorker::Run()
{
    UE_LOG(LogTemp, Log, TEXT("FSocketReceiveWorker Run"));

    // Set a receive timeout for the worker thread
    // This allows the thread to check the StopEvent periodically even if no data is incoming.
    ClientSocket->SetReceiveTimeout(100); // 100 milliseconds timeout

    while (bIsRunning && !StopEvent->IsSignaled())
    {
        // --- Receive Message Size (4 bytes) ---
        uint32 Size = 0;
        int32 BytesRead = 0;
        TArray<uint8> SizeBytes;
        SizeBytes.SetNumUninitialized(4);

        // Use Recv with Peek flag first to check if data is available without removing it
        // Or just use blocking Recv with timeout and handle the timeout.
        // Using blocking Recv with timeout is simpler.
        // Use ClientSocket->Recv(..., ESocketReceiveFlags::Peek) if you want non-blocking check first.

        // Try to receive the 4-byte size prefix
        BytesRead = 0;
        while (BytesRead < 4)
        {
            int32 ThisRead = 0;
            // Recv is blocking and respects the socket timeout (100ms)
            if (!ClientSocket->Recv(SizeBytes.GetData() + BytesRead, 4 - BytesRead, ThisRead))
            {
                // Handle receive failure (timeout or connection error)
                int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
                if (ErrorCode == SE_EWOULDBLOCK || ErrorCode == SE_SOCKET_ERROR) // EWOULDBLOCK can happen with timeouts
                {
                    // Timeout occurred, check stop event and continue loop
                    if (StopEvent->IsSignaled() || !bIsRunning) break; // Exit if stopping
                    continue; // Retry receive
                }
                else // Actual error or connection closed
                {
                    UE_LOG(LogTemp, Error, TEXT("FSocketReceiveWorker: Error receiving message size: %d"), ErrorCode);
                    bIsRunning = false; // Signal to exit loop
                    // TODO: Signal main component/game thread about disconnection
                    break;
                }
            }
            BytesRead += ThisRead;
        }
        if (BytesRead < 4) continue; // Didn't receive the full size prefix (likely connection closed)


        // --- Convert size bytes to integer ---
        Size = FPlatformMisc::BSwap(*(uint32*)SizeBytes.GetData()); // Convert big-endian to host byte order


        // --- Receive Message Payload ---
        TArray<uint8> PayloadBytes;
        PayloadBytes.SetNumUninitialized(Size);
        BytesRead = 0;
         while (BytesRead < Size)
        {
            int32 ThisRead = 0;
            if (!ClientSocket->Recv(PayloadBytes.GetData() + BytesRead, Size - BytesRead, ThisRead))
            {
                // Handle receive failure (timeout or connection error)
                int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
                 if (ErrorCode == SE_EWOULDBLOCK || ErrorCode == SE_SOCKET_ERROR)
                 {
                     if (StopEvent->IsSignaled() || !bIsRunning) break;
                     continue;
                 }
                 else
                 {
                     UE_LOG(LogTemp, Error, TEXT("FSocketReceiveWorker: Error receiving message payload: %d"), ErrorCode);
                     bIsRunning = false;
                     // TODO: Signal main component/game thread about disconnection
                     break;
                 }
            }
            BytesRead += ThisRead;
        }
        if (BytesRead < Size) continue; // Didn't receive the full payload (likely connection closed)


        // --- Process Received Message ---
        // Convert bytes to FString (UTF-8 assumed)
        FString ReceivedJsonString;
        FFileHelper::BufferToString(ReceivedJsonString, PayloadBytes.GetData(), PayloadBytes.Num());

        // Parse JSON
        TSharedPtr<FJsonObject> JsonObject;
        TSharedRef<TJsonReader<TCHAR>> JsonReader = TJsonReaderFactory<TCHAR>::Create(ReceivedJsonString);

        if (FJsonSerializer::Deserialize(JsonReader, JsonObject) && JsonObject.IsValid())
        {
            // Successfully parsed JSON, now extract relevant fields
            FString Action;
            int32 RobotId = -1; // Default or error value

            // Get "action" (string)
            if (!JsonObject->TryGetStringField(TEXT("action"), Action))
            {
                 UE_LOG(LogTemp, Warning, TEXT("FSocketReceiveWorker: Received JSON message missing 'action' field."));
                 continue; // Skip message if missing action
            }

            // Get "robot_id" (integer) - Optional, might not be in all messages
            JsonObject->TryGetNumberField(TEXT("robot_id"), RobotId); // RobotId is optional for some actions

            // Get "parameters" (object) - Optional
            TSharedPtr<FJsonObject> ParametersObject = JsonObject->GetObjectField(TEXT("parameters"));
            // Note: GetObjectField returns nullptr if field is missing or not an object.

            // Create FPythonMessage struct
            FPythonMessage ParsedMessage;
            ParsedMessage.Action = Action;
            ParsedMessage.RobotId = RobotId;
            ParsedMessage.Parameters = ParametersObject; // Keep the shared pointer to the parameters object
            // Add Sequence ID parsing if needed: JsonObject->TryGetNumberField(TEXT("sequence_id"), ParsedMessage.SequenceId);

            // Put the parsed message into the queue for the game thread
            MessageQueue.Enqueue(ParsedMessage);
            UE_LOG(LogTemp, Verbose, TEXT("FSocketReceiveWorker: Parsed and enqueued message with Action: %s"), *Action);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("FSocketReceiveWorker: Failed to parse JSON message: %s"), *ReceivedJsonString);
        }
    }

    // Clean up socket when loop finishes
    if (ClientSocket)
    {
        ClientSocket->Close();
        // The socket object is likely managed by the main component,
        // so avoid deleting it here if it's meant to be reused or managed centrally.
    }

    UE_LOG(LogTemp, Log, TEXT("FSocketReceiveWorker Exit"));
    return 0;
}

void FSocketReceiveWorker::Stop()
{
    bIsRunning = false; // Signal the loop to stop
    StopEvent->Trigger(); // Trigger the event in case the thread is blocked on Recv
    UE_LOG(LogTemp, Log, TEXT("FSocketReceiveWorker Stop requested."));
}

void FSocketReceiveWorker::Exit()
{
    // Clean up thread resources if any
    UE_LOG(LogTemp, Log, TEXT("FSocketReceiveWorker Exited."));
}


// --- UPythonCommServerComponent Implementation ---
UPythonCommServerComponent::UPythonCommServerComponent()
    : TcpListener(nullptr), ClientSocket(nullptr), ReceiveWorker(nullptr), ReceiveThread(nullptr), ReceiveStopEvent(FPlatformProcess::Get0Event())
{
    // Set this component to be ticked
    PrimaryComponentTick.bCanEverTick = true;
}

void UPythonCommServerComponent::BeginPlay()
{
    Super::BeginPlay();

    StartServer(); // Start the server when the game starts
}

void UPythonCommServerComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);

    StopServer(); // Stop the server when the game ends
}

bool UPythonCommServerComponent::StartServer()
{
    if (TcpListener)
    {
        UE_LOG(LogTemp, Warning, TEXT("Python Comm Server already started."));
        return false;
    }

    FIPv4Address Addr;
    FIPv4Address::Parse(ListenIP, Addr);

    FIPv4Endpoint Endpoint(Addr, ListenPort);

    // Create the TCP Listener
    TcpListener = new FTcpSocketListener(Endpoint, false); // false = non-threaded listener

    // Bind the connection accepted delegate
    // The listener will call this function when a new connection is accepted
    TcpListener->OnForConnectionAccepted().BindUObject(this, &UPythonCommServerComponent::OnSocketConnectionAccepted);

    // Start the listener
    if (TcpListener->Start())
    {
        UE_LOG(LogTemp, Log, TEXT("Python Comm Server listening on %s:%d"), *ListenIP, ListenPort);
        return true;
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to start Python Comm Server on %s:%d"), *ListenIP, ListenPort);
        delete TcpListener;
        TcpListener = nullptr;
        return false;
    }
}

void UPythonCommServerComponent::StopServer()
{
    if (TcpListener)
    {
        // Stop the listener
        TcpListener->Stop();
        delete TcpListener;
        TcpListener = nullptr;
        UE_LOG(LogTemp, Log, TEXT("Python Comm Server stopped."));
    }

    // Ensure client connection and worker thread are stopped
    if (ReceiveWorker)
    {
        ReceiveWorker->Stop(); // Signal the worker thread to stop
        // Optional: Wait for the thread to finish (join)
        if (ReceiveThread)
        {
            ReceiveThread->WaitForCompletion();
            delete ReceiveThread;
            ReceiveThread = nullptr;
        }
        delete ReceiveWorker;
        ReceiveWorker = nullptr;
         if (ReceiveStopEvent) {
             FPlatformProcess::FreeEvent(ReceiveStopEvent);
             ReceiveStopEvent = nullptr;
         }
    }

    if (ClientSocket)
    {
        ClientSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ClientSocket);
        ClientSocket = nullptr;
        UE_LOG(LogTemp, Log, TEXT("Python client connection closed."));
         OnClientDisconnected.Broadcast(); // Trigger disconnected event
    }
}

bool UPythonCommServerComponent::OnSocketConnectionAccepted(FSocket* InClientSocket, const FIPv4Endpoint& ClientEndpoint)
{
    // This function is called by the listener thread when a new client connects.
    // We need to handle this connection on the game thread or pass it to a worker.

    UE_LOG(LogTemp, Log, TEXT("Python client connected from %s"), *ClientEndpoint.ToString());

    // For simplicity, accept only one connection at a time.
    if (ClientSocket && ClientSocket->GetConnectionState() == SCS_Connected)
    {
        UE_LOG(LogTemp, Warning, TEXT("Another client is already connected. Rejecting new connection."));
        // Close the new socket immediately
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(InClientSocket);
        return false; // Tell the listener not to keep this socket
    }

    // Accept the new connection
    ClientSocket = InClientSocket;

    // Signal the game thread about the connection (optional, could use a queue)
    // Or trigger a delegate directly if it's safe / handled on game thread
    OnClientConnected.Broadcast(); // Trigger connected event

    // Start a new worker thread to receive data from this client socket
    ReceiveStopEvent = FPlatformProcess::Get0Event(); // Create the event object
    ReceiveWorker = new FSocketReceiveWorker(ClientSocket, ReceivedMessageQueue, ReceiveStopEvent);
    ReceiveThread = FRunnableThread::Create(ReceiveWorker, TEXT("PythonCommReceiveThread"));

    return true; // Tell the listener to keep this socket
}

void UPythonCommServerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickType);

    // Process received messages from the queue on the game thread
    ProcessReceivedMessages();
}

void UPythonCommServerComponent::ProcessReceivedMessages()
{
    FPythonMessage Message;
    // Dequeue messages. Use Peek to check without removing if needed, or Dequeue with timeout.
    // DequeueAll is efficient if many messages might arrive per tick.
    while (ReceivedMessageQueue.Dequeue(Message))
    {
        // Process the received message on the game thread
        UE_LOG(LogTemp, Log, TEXT("Processing message on Game Thread: Action = %s, RobotId = %d"), *Message.Action, Message.RobotId);

        // --- Dispatch Actions ---
        if (Message.Action == TEXT("set_robot_pose"))
        {
            // Handle Set Robot Pose command
            // Find the target robot actor and update its pose
            AActor** RobotActorPtr = RegisteredRobots.Find(Message.RobotId);
            if (RobotActorPtr && *RobotActorPtr)
            {
                // Cast the actor to your specific Robot Actor class if needed
                // AYourRobotBaseActor* RobotActor = Cast<AYourRobotBaseActor>(*RobotActorPtr);
                // if (RobotActor) {
                //     RobotActor->SetRobotPoseFromJSON(Message.Parameters); // Call BlueprintCallable or C++ function
                // }
                // For simplicity, just log or call a generic function if your base actor has one
                 UE_LOG(LogTemp, Log, TEXT("Received set_robot_pose for Robot %d. Dispatching..."), Message.RobotId);
                 // You would implement dispatch logic here, potentially calling a function on the Actor:
                 // UGameplayStatics::CallFunctionByName(*RobotActorPtr, TEXT("SetRobotPoseFromJSON"), params); // Generic function call
                 // Or trigger a Blueprint event if using Blueprint
                 OnMessageReceived.Broadcast(Message); // Let Blueprint handle dispatch via this event

            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Received set_robot_pose for unknown or unregistered Robot ID: %d"), Message.RobotId);
            }
        }
        else if (Message.Action == TEXT("welding_visual_command"))
        {
            // Handle Welding Visual Command
            AActor** RobotActorPtr = RegisteredRobots.Find(Message.RobotId);
            if (RobotActorPtr && *RobotActorPtr)
            {
                 UE_LOG(LogTemp, Log, TEXT("Received welding_visual_command for Robot %d. Dispatching..."), Message.RobotId);
                 // Call a function on the Actor or trigger an event:
                 OnMessageReceived.Broadcast(Message); // Let Blueprint handle dispatch via this event
            }
             else
            {
                UE_LOG(LogTemp, Warning, TEXT("Received welding_visual_command for unknown or unregistered Robot ID: %d"), Message.RobotId);
            }

        }
        else if (Message.Action == TEXT("run_simulation"))
        {
             // Handle Run Simulation command
             UE_LOG(LogTemp, Log, TEXT("Received run_simulation command."));
             // Implement simulation logic here or delegate to another component
             // After simulation, send result back using SendJsonResponse
             OnMessageReceived.Broadcast(Message); // Let Blueprint or another C++ component handle simulation
        }
        else if (Message.Action == TEXT("get_sim2real_ark_situation"))
        {
             // Handle Get Sim2Real command
             UE_LOG(LogTemp, Log, TEXT("Received get_sim2real_ark_situation command."));
             // Implement data generation logic
             // Send data back using SendJsonResponse
             OnMessageReceived.Broadcast(Message); // Let Blueprint or another C++ component handle generating data
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Received unknown action: %s"), *Message.Action);
            // Potentially send an error response back to Python
        }
    }
}


bool UPythonCommServerComponent::SendFramedData(const FString& DataToSend)
{
    if (!ClientSocket || ClientSocket->GetConnectionState() != SCS_Connected)
    {
        UE_LOG(LogTemp, Warning, TEXT("Cannot send data, client socket not connected."));
        return false;
    }

    TArray<uint8> PayloadBytes;
    FTCHARToUTF8 Converter(*DataToSend);
    PayloadBytes.SetNum(Converter.Length());
    FMemory::Memcpy(PayloadBytes.GetData(), Converter.Get(), Converter.Length());

    uint32 Size = PayloadBytes.Num();
    TArray<uint8> SizeBytes;
    SizeBytes.SetNumUninitialized(4);
    // Convert size to big-endian
    uint32 BigEndianSize = FPlatformMisc::BSwap(Size);
    FMemory::Memcpy(SizeBytes.GetData(), &BigEndianSize, 4);


    int32 BytesSent = 0;
    bool Success = false;

    // Use a lock if SendFramedData can be called from multiple threads
    // (Unlikely in this Tick-based model, but good practice if logic changes)
    // with SendLock: // Define SendLock if needed
    {
        // Send size prefix
        Success = ClientSocket->Send(SizeBytes.GetData(), SizeBytes.Num(), BytesSent);
        if (Success && BytesSent == SizeBytes.Num())
        {
            // Send payload
            Success = ClientSocket->Send(PayloadBytes.GetData(), PayloadBytes.Num(), BytesSent);
            if (Success && BytesSent == PayloadBytes.Num())
            {
                 UE_LOG(LogTemp, Verbose, TEXT("Sent %d bytes payload."), Size);
                 return true; // Successfully sent size and payload
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("Failed to send payload bytes. Sent: %d / %d"), BytesSent, Size);
                 Success = false; // Ensure Success is false on payload send failure
            }
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to send size bytes. Sent: %d / 4"), BytesSent);
             Success = false; // Ensure Success is false on size send failure
        }
    }

    if (!Success)
    {
        // Handle send failure (likely connection error)
        int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
        UE_LOG(LogTemp, Error, TEXT("SendFramedData error: %d. Disconnecting client."), ErrorCode);
        // TODO: Signal main component/game thread about disconnection
        // Calling Disconnect here directly might not be safe if this is not on the game thread.
        // Need a way to safely signal disconnection to the game thread.
        // For simplicity in this component, let's assume SendFramedData is called on the game thread.
        // If called from worker, add a queue/delegate to signal game thread.
        StopServer(); // Simple approach: Stop everything on send failure
    }

    return Success;
}

bool UPythonCommServerComponent::SendJsonResponse(int32 ClientConnectionId, const FString& Status, const FString& Message, const TSharedPtr<FJsonObject>& Data)
{
    if (!ClientSocket || ClientSocket->GetConnectionState() != SCS_Connected)
    {
         UE_LOG(LogTemp, Warning, TEXT("Cannot send JSON response, client socket not connected."));
         return false;
    }
    // NOTE: ClientConnectionId is ignored here in the single-client model.
    // In a multi-client setup, you'd use this ID to find the correct FSocket.

    TSharedPtr<FJsonObject> ResponseObject = MakeShareable(new FJsonObject());
    ResponseObject->SetStringField(TEXT("status"), Status);
    ResponseObject->SetStringField(TEXT("message"), Message);
    if (Data.IsValid())
    {
        ResponseObject->SetObjectField(TEXT("data"), Data); // Or SetField if Data is not always an object
    }
    // TODO: Include sequence_id from the original request if the protocol requires matching responses!

    FString ResponseString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&ResponseString);
    FJsonSerializer::Serialize(ResponseObject.ToSharedRef(), Writer);

    UE_LOG(LogTemp, Log, TEXT("Sending JSON Response: %s"), *ResponseString);
    return SendFramedData(ResponseString);
}


void UPythonCommServerComponent::RegisterRobotActor(int32 RobotId, AActor* RobotActor)
{
    if (!RobotActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("Attempted to register null Robot Actor for ID %d."), RobotId);
        return;
    }
    if (RegisteredRobots.Contains(RobotId))
    {
        UE_LOG(LogTemp, Warning, TEXT("Robot ID %d is already registered. Overwriting."), RobotId);
    }
    RegisteredRobots.Add(RobotId, RobotActor);
    UE_LOG(LogTemp, Log, TEXT("Registered Robot Actor '%s' with ID %d."), *RobotActor->GetName(), RobotId);
}