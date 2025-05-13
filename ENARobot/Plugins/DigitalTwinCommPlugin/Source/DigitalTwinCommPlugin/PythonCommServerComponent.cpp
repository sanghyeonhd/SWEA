// Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/PythonCommServerComponent.cpp

#include "PythonCommServerComponent.h" // 자체 헤더 포함
#include "Sockets.h" // FSocket, ISocketSubsystem
#include "Networking.h" // FIPv4Endpoint, FIPv4Address
#include "TimerManager.h" // FTimerManager (필요시)
#include "Json.h" // FJsonReader, FJsonSerializer
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "HAL/RunnableThread.h" // FRunnableThread
#include "HAL/PlatformProcess.h" // FPlatformProcess::Get0Event, FreeEvent
#include "Misc/SingleThreadEvent.h" // FSingleThreadEvent (대신 FEvent 사용)
#include "Misc/FileHelper.h" // FFileHelper::BufferToString
#include "Logging/MessageLog.h" // FMessageLog (선택적)
#include "Containers/Array.h"
#include "Containers/StringConv.h" // FCharToUTF8

// UE 로깅 매크로 사용 (모듈/플러그인 이름에 맞는 LogCategory 정의 필요)
// 예시 LogCategory: LogDigitalTwinComm
DEFINE_LOG_CATEGORY_STATIC(LogDigitalTwinComm, Log, All);


// --- FSocketReceiveWorker Implementation ---
// 백그라운드 스레드에서 실행되어 소켓 수신을 담당합니다.
FSocketReceiveWorker::FSocketReceiveWorker(FSocket* InSocket, TQueue<FPythonMessage, EQueueMode::Mpsc>& InMessageQueue, FEvent* InStopEvent)
    : ClientSocket(InSocket), MessageQueue(InMessageQueue), StopEvent(InStopEvent)
{
    bIsRunning = true;
}

FSocketReceiveWorker::~FSocketReceiveWorker()
{
     // 리소스 정리
     ClientSocket = nullptr; // 소켓은 컴포넌트에서 관리하므로 여기서 삭제하지 않음
     StopEvent = nullptr; // 이벤트는 컴포넌트에서 관리하므로 여기서 삭제하지 않음
}


bool FSocketReceiveWorker::Init()
{
    // 스레드 시작 시 호출됩니다. 초기화 성공 시 true 반환.
    UE_LOG(LogDigitalTwinComm, Log, TEXT("FSocketReceiveWorker Init"));
    // 소켓이 유효한지 확인
    if (!ClientSocket || ClientSocket->GetConnectionState() != SCS_Connected)
    {
        UE_LOG(LogDigitalTwinComm, Error, TEXT("FSocketReceiveWorker failed to initialize: Invalid client socket."));
        return false;
    }
     // 수신 타임아웃 설정 (Run 함수 내에서 설정하는 것이 더 안전할 수 있음)
    // ClientSocket->SetReceiveTimeout(100); // 예시: 100ms 타임아웃
    return true;
}

uint32 FSocketReceiveWorker::Run()
{
    // 스레드의 메인 실행 루프입니다. Stop()이 호출되거나 오류 발생 시 종료됩니다.
    UE_LOG(LogDigitalTwinComm, Log, TEXT("FSocketReceiveWorker Run"));

    // 소켓 수신 타임아웃 설정
    // Recv 함수가 이 시간만큼 블록했다가 타임아웃 오류(SE_EWOULDBLOCK)를 발생시킵니다.
    // 이를 통해 루프가 StopEvent를 주기적으로 체크할 수 있습니다.
    int32 ReceiveTimeoutMs = 100; // 100ms
    ClientSocket->SetReceiveTimeout(ReceiveTimeoutMs);


    TArray<uint8> SizeBytes; // 4바이트 크기 정보 저장용 버퍼
    SizeBytes.SetNumUninitialized(4);

    while (bIsRunning && !StopEvent->IsSignaled())
    {
        // --- 1. 메시지 크기 (4바이트) 수신 ---
        int32 BytesRead = 0;
        // ReceiveExactly 헬퍼 함수를 사용하면 안전하게 지정된 바이트 수만큼 읽을 수 있습니다.
        // 하지만 타임아웃 처리를 위해 Recv를 직접 사용하는 것이 더 좋습니다.
        while (BytesRead < 4)
        {
            int32 ThisRead = 0;
            // Recv 함수는 지정된 타임아웃 동안 데이터가 도착할 때까지 블록합니다.
            if (!ClientSocket->Recv(SizeBytes.GetData() + BytesRead, 4 - BytesRead, ThisRead))
            {
                // 수신 실패 (타임아웃 또는 연결 오류) 처리
                int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
                if (ErrorCode == SE_EWOULDBLOCK || ErrorCode == SE_SOCKET_ERROR) // EWOULDBLOCK == 타임아웃
                {
                    // 타임아웃 발생: 데이터를 받지 못했지만 연결은 유지됨. Stop Event 체크 후 계속 대기.
                    if (StopEvent->IsSignaled() || !bIsRunning) break; // 중지 신호가 왔으면 루프 종료
                    continue; // 데이터가 없었으므로 다음 루프에서 다시 시도 (타임아웃 만큼 대기 후)
                }
                else // 실제 연결 오류 또는 연결 종료
                {
                    UE_LOG(LogDigitalTwinComm, Error, TEXT("FSocketReceiveWorker: Fatal error receiving message size: %d"), ErrorCode);
                    bIsRunning = false; // 치명적 오류 발생, 스레드 종료 신호
                    // TODO: 게임 스레드에 연결 끊김을 알리는 로직 추가 (Delegate or Queue)
                    break; // 루프 종료
                }
            }
            BytesRead += ThisRead; // 성공적으로 읽은 바이트 수 누적
        }
        if (BytesRead < 4) continue; // 4바이트 크기 정보를 완전히 읽지 못했으면 (예: 연결 끊김) 이번 루프 건너뛰기


        // --- 2. 크기 정보 변환 ---
        // Python에서 big-endian으로 보냈으므로, 호스트 시스템의 엔디안에 맞게 바이트 스왑합니다.
        uint32 Size = FPlatformMisc::BSwap(*(uint32*)SizeBytes.GetData());
        // UE_LOG(LogDigitalTwinComm, Verbose, TEXT("FSocketReceiveWorker: Expecting %d bytes payload."), Size);

        if (Size == 0)
        {
            UE_LOG(LogDigitalTwinComm, Warning, TEXT("FSocketReceiveWorker: Received message size 0. Skipping payload."));
            continue; // 크기가 0인 메시지는 페이로드 수신 없이 건너뜁니다.
        }
        if (Size > 10 * 1024 * 1024) // 예시: 너무 큰 메시지 필터링 (DoS 공격 방지 등)
        {
            UE_LOG(LogDigitalTwinComm, Error, TEXT("FSocketReceiveWorker: Received excessively large message size %d. Potential error or attack. Disconnecting."), Size);
            bIsRunning = false;
            // TODO: 게임 스레드에 연결 끊김 알림
            break; // 루프 종료
        }


        // --- 3. 메시지 페이로드 수신 ---
        TArray<uint8> PayloadBytes; // 페이로드 저장용 버퍼
        PayloadBytes.SetNumUninitialized(Size);
        BytesRead = 0;

        while (BytesRead < Size)
        {
            int32 ThisRead = 0;
            // 남은 크기(Size - BytesRead) 만큼 수신 시도. 역시 타임아웃 적용.
            if (!ClientSocket->Recv(PayloadBytes.GetData() + BytesRead, Size - BytesRead, ThisRead))
            {
                int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
                 if (ErrorCode == SE_EWOULDBLOCK || ErrorCode == SE_SOCKET_ERROR)
                 {
                     // 타임아웃 발생: 페이로드를 다 받지 못했지만 계속 대기.
                     if (StopEvent->IsSignaled() || !bIsRunning) break;
                     continue;
                 }
                 else // 실제 연결 오류 또는 연결 종료
                 {
                     UE_LOG(LogDigitalTwinComm, Error, TEXT("FSocketReceiveWorker: Fatal error receiving message payload: %d"), ErrorCode);
                     bIsRunning = false; // 치명적 오류, 스레드 종료
                     // TODO: 게임 스레드에 연결 끊김 알림
                     break; // 루프 종료
                 }
            }
            BytesRead += ThisRead; // 성공적으로 읽은 바이트 수 누적
        }
        if (BytesRead < Size) continue; // 페이로드를 완전히 읽지 못했으면 (예: 연결 끊김) 이번 루프 건너뛰기


        // --- 4. 수신된 페이로드 처리 ---
        // 바이트를 FString으로 변환 (UTF-8 디코딩)
        FString ReceivedJsonString;
        FFileHelper::BufferToString(ReceivedJsonString, PayloadBytes.GetData(), PayloadBytes.Num());
        // UE_LOG(LogDigitalTwinComm, VeryVerbose, TEXT("FSocketReceiveWorker: Received JSON string: %s"), *ReceivedJsonString); // 너무 상세할 수 있음

        // JSON 파싱
        TSharedPtr<FJsonObject> JsonObject; // 파싱 결과 저장용 FJsonObject
        TSharedRef<TJsonReader<TCHAR>> JsonReader = TJsonReaderFactory<TCHAR>::Create(ReceivedJsonString); // JSON Reader 생성

        if (FJsonSerializer::Deserialize(JsonReader, JsonObject) && JsonObject.IsValid()) // 파싱 시도 및 결과 확인
        {
            // JSON 파싱 성공! 메시지 내용 추출
            FPythonMessage ParsedMessage; // 파싱 결과를 담을 USTRUCT (또는 일반 struct)

            // 'action' 필드 추출 (필수)
            if (!JsonObject->TryGetStringField(TEXT("action"), ParsedMessage.Action))
            {
                 UE_LOG(LogDigitalTwinComm, Warning, TEXT("FSocketReceiveWorker: Received JSON message missing 'action' field. Skipping message. Raw: %s"), *ReceivedJsonString);
                 continue; // 'action' 필드가 없으면 유효하지 않은 메시지로 간주하고 건너뜁니다.
            }

            // 'robot_id' 필드 추출 (선택적, 정수)
            // TryGetNumberField는 필드가 없거나 숫자가 아니면 false 반환하고 값을 변경하지 않음
            if (!JsonObject->TryGetNumberField(TEXT("robot_id"), ParsedMessage.RobotId))
            {
                ParsedMessage.RobotId = -1; // 'robot_id'가 없거나 유효하지 않으면 기본값 -1 설정
            }

            // 'parameters' 필드 추출 (선택적, JSON Object)
            // GetObjectField는 필드가 없거나 오브젝트가 아니면 nullptr 반환
            ParsedMessage.Parameters = JsonObject->GetObjectField(TEXT("parameters"));


            // 'sequence_id' 필드 추출 (선택적, 응답 추적 필요시)
            // JsonObject->TryGetNumberField(TEXT("sequence_id"), ParsedMessage.SequenceId);


            // 파싱된 메시지를 게임 스레드로 전달하기 위한 큐에 넣습니다.
            MessageQueue.Enqueue(ParsedMessage);
            UE_LOG(LogDigitalTwinComm, Verbose, TEXT("FSocketReceiveWorker: Parsed and enqueued message. Action: %s, RobotId: %d"), *ParsedMessage.Action, ParsedMessage.RobotId);
        }
        else
        {
            // JSON 파싱 실패
            UE_LOG(LogDigitalTwinComm, Warning, TEXT("FSocketReceiveWorker: Failed to parse JSON message. Raw data: %s"), *ReceivedJsonString);
            // 유효하지 않은 메시지 처리 로직 추가 필요 (예: 오류 응답 전송)
        }
    } // while 루프 종료

    // 루프 종료 후 정리 작업
    // 소켓은 컴포넌트에서 관리하므로 여기서 삭제하지 않음.
    // 클라이언트 연결 끊김 처리 로직은 _receive_framed_json 또는 해당 함수를 호출한 곳에서 수행.

    UE_LOG(LogDigitalTwinComm, Log, TEXT("FSocketReceiveWorker Exit"));
    return 0; // Run 함수는 항상 0을 반환 (완료)
}

void FSocketReceiveWorker::Stop()
{
    // 스레드 중지 신호를 설정합니다.
    bIsRunning = false;
    // Recv 함수가 블록되어 있을 경우, StopEvent를 Trigger하여 블록킹을 해제하고 스레드가 종료되도록 합니다.
    if (StopEvent)
    {
         StopEvent->Trigger();
         // UE_LOG(LogDigitalTwinComm, Log, TEXT("FSocketReceiveWorker StopEvent triggered."));
    }
    else
    {
         UE_LOG(LogDigitalTwinComm, Warning, TEXT("FSocketReceiveWorker StopEvent is null. Cannot trigger."));
    }
}

void FSocketReceiveWorker::Exit()
{
    // 스레드가 완전히 종료된 후 호출됩니다.
    UE_LOG(LogDigitalTwinComm, Log, TEXT("FSocketReceiveWorker Exited."));
}


// --- UPythonCommServerComponent Implementation ---
UPythonCommServerComponent::UPythonCommServerComponent()
    // 멤버 변수들 기본값 초기화
    : TcpListener(nullptr)
    , ClientSocket(nullptr)
    , ReceiveWorker(nullptr)
    , ReceiveThread(nullptr)
    , ReceiveStopEvent(nullptr) // BeginPlay에서 생성
    // ReceivedMessageQueue는 생성자에서 기본값으로 초기화됨 (Mpsc 큐)
{
    // 컴포넌트를 Tick 가능하도록 설정 (매 프레임 TickComponent 호출)
    // 메시지 큐 처리를 위해 필요합니다.
    PrimaryComponentTick.bCanEverTick = true;
}

// 소멸자: 객체 파괴 시 정리 작업
UPythonCommServerComponent::~UPythonCommServerComponent()
{
    // Ensure server and threads are stopped when the component is destroyed
    StopServer(); // StopServer 함수 호출
}


// 게임 시작 시 호출
void UPythonCommServerComponent::BeginPlay()
{
    Super::BeginPlay();

    // 컴포넌트가 스폰될 때 서버 시작
    StartServer();
}

// 게임 종료 또는 액터 언로드 시 호출
void UPythonCommServerComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // 컴포넌트가 파괴될 때 서버 중지
    StopServer();

    Super::EndPlay(EndPlayReason);
}

// 서버 시작 함수
bool UPythonCommServerComponent::StartServer()
{
    if (TcpListener)
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Python Comm Server already started."));
        return false;
    }

    // 리스닝 IP와 포트 파싱
    FIPv4Address Addr;
    if (!FIPv4Address::Parse(ListenIP, Addr))
    {
        UE_LOG(LogDigitalTwinComm, Error, TEXT("Invalid ListenIP format: %s"), *ListenIP);
        return false;
    }
    FIPv4Endpoint Endpoint(Addr, ListenPort);

    // TCP Listener 생성
    // false 인자는 리스너 자체는 스레드를 생성하지 않음을 의미.
    // 연결 수락 시 처리 로직(OnForConnectionAccepted 델리게이트)에서 별도 스레드를 시작해야 합니다.
    TcpListener = new FTcpSocketListener(Endpoint, false);

    // 연결 수락 시 호출될 델리게이트 바인딩
    // 새 클라이언트 연결이 수락되면 OnSocketConnectionAccepted 함수가 호출됩니다.
    TcpListener->OnForConnectionAccepted().BindUObject(this, &UPythonCommServerComponent::OnSocketConnectionAccepted);

    // 리스너 시작
    if (TcpListener->Start())
    {
        UE_LOG(LogDigitalTwinComm, Log, TEXT("Python Comm Server listening on %s:%d"), *ListenIP, ListenPort);
        return true;
    }
    else
    {
        // 리스너 시작 실패
        int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
        UE_LOG(LogDigitalTwinComm, Error, TEXT("Failed to start Python Comm Server on %s:%d. Error: %d"), *ListenIP, ListenPort, ErrorCode);
        delete TcpListener;
        TcpListener = nullptr;
        return false;
    }
}

// 서버 중지 함수
void UPythonCommServerComponent::StopServer()
{
    // 클라이언트 수신 스레드 먼저 중지
    if (ReceiveWorker)
    {
        ReceiveWorker->Stop(); // 스레드에 중지 신호 전달
        // 스레드가 안전하게 종료될 때까지 기다립니다 (Join).
        // WaitForCompletion은 스레드 핸들이 유효할 경우에만 호출.
        if (ReceiveThread)
        {
            ReceiveThread->WaitForCompletion();
            // 스레드 객체와 스레드 핸들 삭제
            delete ReceiveThread;
            ReceiveThread = nullptr;
        }
        // Worker 객체 삭제 (스레드 종료 후)
        delete ReceiveWorker;
        ReceiveWorker = nullptr;
        // 이벤트 객체 해제
         if (ReceiveStopEvent) {
             FPlatformProcess::FreeEvent(ReceiveStopEvent);
             ReceiveStopEvent = nullptr;
         }
        UE_LOG(LogDigitalTwinComm, Log, TEXT("Python client receive worker stopped."));
    }

    // 클라이언트 소켓 연결 해제
    if (ClientSocket)
    {
        ClientSocket->Close();
        // 소켓 서브시스템을 통해 소켓 객체 파괴
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ClientSocket);
        ClientSocket = nullptr;
        UE_LOG(LogDigitalTwinComm, Log, TEXT("Python client connection closed."));
         // 클라이언트 연결 끊김 델리게이트 호출
         OnClientDisconnected.Broadcast();
    }

    // TCP 리스너 중지
    if (TcpListener)
    {
        TcpListener->Stop();
        delete TcpListener;
        TcpListener = nullptr;
        UE_LOG(LogDigitalTwinComm, Log, TEXT("Python Comm Server stopped."));
    }
}

// 새 클라이언트 연결이 수락되면 Listener 스레드에서 호출되는 함수
bool UPythonCommServerComponent::OnSocketConnectionAccepted(FSocket* InClientSocket, const FIPv4Endpoint& ClientEndpoint)
{
    // 이 함수는 Listener 스레드에서 실행되므로, 게임 스레드 안전성에 유의해야 합니다.
    // 여기서는 단순히 연결을 수락하고, 데이터 수신 처리는 별도 Worker 스레드에 위임합니다.

    UE_LOG(LogDigitalTwinComm, Log, TEXT("Python client connected from %s"), *ClientEndpoint.ToString());

    // 예시에서는 한 번에 하나의 클라이언트만 허용합니다.
    // 기존 클라이언트 소켓이 이미 연결되어 있다면 새 연결을 거부합니다.
    if (ClientSocket && ClientSocket->GetConnectionState() == SCS_Connected)
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Another client is already connected. Rejecting new connection from %s."), *ClientEndpoint.ToString());
        // 새 소켓을 닫고 파괴합니다.
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(InClientSocket);
        return false; // Listener에게 이 소켓을 관리하지 말라고 알립니다.
    }

    // 기존 연결이 없거나 끊어졌다면, 새 연결을 수락합니다.
    // 이전 Worker 스레드가 남아있다면 정리해야 합니다.
    if (ReceiveWorker)
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Previous receive worker still exists. Stopping it."));
        StopServer(); // 간단하게 기존 서버 관련 모두 중지 후 새로 시작
        // 또는 ReceiveWorker와 ClientSocket만 정리하는 로직 구현
    }


    // 새 클라이언트 소켓을 저장합니다.
    ClientSocket = InClientSocket;

    // 게임 스레드에 클라이언트 연결됨을 알립니다 (Tick에서 큐를 통해 처리할 수도 있음)
    // 델리게이트 호출은 안전성에 유의해야 하지만, OnForConnectionAccepted는 특정 스레드에서 호출되므로
    // 안전하게 게임 스레드에서 처리되도록 보장하는 메커니즘이 필요할 수 있습니다.
    // TQueue나 TaskGraph를 사용하여 게임 스레드에 작업을 Post하는 것이 더 안전합니다.
    // 예시에서는 Broadcast 호출 후 Tick에서 큐 처리를 통해 간접적으로 이벤트를 발생시킵니다.
    // OnClientConnected.Broadcast(); // Tick에서 ProcessReceivedMessages 호출 시 처리하도록 변경


    // 새 클라이언트 소켓으로부터 데이터 수신을 담당할 Worker 스레드를 생성하고 시작합니다.
    ReceiveStopEvent = FPlatformProcess::Get0Event(); // 이벤트 객체 생성 (Worker 스레드 중지용)
    ReceiveWorker = new FSocketReceiveWorker(ClientSocket, ReceivedMessageQueue, ReceiveStopEvent);
    ReceiveThread = FRunnableThread::Create(ReceiveWorker, TEXT("PythonCommReceiveThread")); // 스레드 생성 및 시작

    UE_LOG(LogDigitalTwinComm, Log, TEXT("Accepted new client connection and started receive worker."));
    return true; // Listener에게 이 소켓을 관리하라고 알립니다.
}

// 매 프레임 호출되어 메시지 큐를 확인하고 메시지를 처리합니다.
void UPythonCommServerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickType);

    // 수신 스레드에서 받은 메시지들을 게임 스레드에서 처리합니다.
    ProcessReceivedMessages();

    // 클라이언트 연결 상태를 주기적으로 확인할 수도 있습니다.
    // if (ClientSocket && ClientSocket->GetConnectionState() == SCS_NotConnected)
    // {
    //      UE_LOG(LogDigitalTwinComm, Warning, TEXT("Client socket is no longer connected."));
    //      StopServer(); // 클라이언트 연결 끊김 감지 시 서버 중지 (예시)
    // }
}

// 수신 큐에서 메시지를 가져와 게임 스레드에서 처리하는 함수
void UPythonCommServerComponent::ProcessReceivedMessages()
{
    FPythonMessage Message;
    // ReceivedMessageQueue.Dequeue(Message)는 큐가 비어 있으면 false 반환
    // ReceivedMessageQueue.Dequeue(Message, 0)는 논블록킹.
    // TickComponent에서 호출되므로, 논블록킹으로 모든 메시지를 가져와 처리해야 합니다.
    while (ReceivedMessageQueue.Dequeue(Message))
    {
        // 큐에서 메시지를 성공적으로 가져왔습니다.
        // 이 메시지는 이미 Worker 스레드에서 JSON 파싱까지 완료된 상태입니다.
        UE_LOG(LogDigitalTwinComm, Log, TEXT("Processing message on Game Thread: Action = %s, RobotId = %d"), *Message.Action, Message.RobotId);

        // --- 액션 디스패치 ---
        // 수신된 메시지의 'action' 필드를 기반으로 적절한 처리 로직을 호출합니다.
        // 블루프린트에서 이 메시지를 처리하도록 Delegate를 호출하는 것이 유연합니다.
        // 또는 C++에서 직접 처리 함수를 호출할 수도 있습니다.

        // FPythonMessage 구조체는 Parameters TSharedPtr를 가지고 있으므로,
        // 델리게이트를 통해 블루프린트로 전달하고 블루프린트에서 GetObjectField 등을 사용하여
        // Parameters 내 상세 파라미터들을 추출하도록 할 수 있습니다.

        // OnMessageReceived 델리게이트 호출 (블루프린트 이벤트 트리거)
        OnMessageReceived.Broadcast(Message);

        // --- C++에서 직접 액션 디스패치하는 예시 ---
        /*
        if (Message.Action == TEXT("set_robot_pose"))
        {
             // 로봇 액터를 찾고 포즈 설정 함수 호출
             AActor** RobotActorPtr = RegisteredRobots.Find(Message.RobotId);
             if (RobotActorPtr && *RobotActorPtr)
             {
                 ARobotBaseActor* RobotActor = Cast<ARobotBaseActor>(*RobotActorPtr);
                 if (RobotActor) {
                     RobotActor->SetRobotPoseFromJSON(Message.Parameters); // C++ 함수 호출
                 } else { UE_LOG(LogDigitalTwinComm, Warning, TEXT("Registered Actor for ID %d is not an ARobotBaseActor."), Message.RobotId); }
             } else { UE_LOG(LogDigitalTwinComm, Warning, TEXT("Received set_robot_pose for unknown Robot ID: %d"), Message.RobotId); }
        }
        else if (Message.Action == TEXT("welding_visual_command"))
        {
             // 로봇 액터를 찾고 시각 명령 처리 함수 호출
             AActor** RobotActorPtr = RegisteredRobots.Find(Message.RobotId);
             if (RobotActorPtr && *RobotActorPtr)
             {
                 ARobotBaseActor* RobotActor = Cast<ARobotBaseActor>(*RobotActorPtr);
                 if (RobotActor) {
                     RobotActor->HandleWeldingVisualCommand(Message.Parameters); // C++ 함수 호출
                 } else { UE_LOG(LogDigitalTwinComm, Warning, TEXT("Registered Actor for ID %d is not an ARobotBaseActor."), Message.RobotId); }
             } else { UE_LOG(LogDigitalTwinComm, Warning, TEXT("Received welding_visual_command for unknown Robot ID: %d"), Message.RobotId); }
        }
        // ... 다른 액션 처리 (run_simulation, get_sim2real_ark_situation 등)
        else {
            UE_LOG(LogDigitalTwinComm, Warning, TEXT("Received unknown action: %s"), *Message.Action);
        }
        */
    } // while (Dequeue) 루프 종료
}


// Python 클라이언트로 데이터 전송 (4바이트 길이 접두사 + 데이터)
bool UPythonCommServerComponent::SendFramedData(const FString& DataToSend)
{
    if (!ClientSocket || ClientSocket->GetConnectionState() != SCS_Connected)
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Cannot send data, client socket not connected."));
        return false;
    }

    // FString을 UTF-8 바이트 배열로 변환
    FTCHARToUTF8 Converter(*DataToSend);
    TArray<uint8> PayloadBytes;
    PayloadBytes.SetNumUninitialized(Converter.Length());
    FMemory::Memcpy(PayloadBytes.GetData(), Converter.Get(), Converter.Length());

    // 페이로드 크기 계산 및 Big-endian 4바이트로 변환
    uint32 Size = PayloadBytes.Num();
    TArray<uint8> SizeBytes;
    SizeBytes.SetNumUninitialized(4);
    uint32 BigEndianSize = FPlatformMisc::BSwap(Size); // 바이트 스왑
    FMemory::Memcpy(SizeBytes.GetData(), &BigEndianSize, 4);


    int32 BytesSent = 0;
    bool Success = false;

    // 소켓 송신은 스레드 안전하지 않으므로, 필요시 락 사용.
    // TickComponent에서만 SendFramedData를 호출한다면 락이 필요 없을 수 있음.
    // 예시에서는 Tick에서만 호출한다고 가정.
    { // with SendLock: // Define and use a lock if SendFramedData is called from multiple threads
        // 1. 크기 정보 전송
        Success = ClientSocket->Send(SizeBytes.GetData(), SizeBytes.Num(), BytesSent);
        if (Success && BytesSent == SizeBytes.Num())
        {
            // 2. 페이로드 전송
            Success = ClientSocket->Send(PayloadBytes.GetData(), PayloadBytes.Num(), BytesSent);
            if (Success && BytesSent == PayloadBytes.Num())
            {
                 UE_LOG(LogDigitalTwinComm, Verbose, TEXT("Sent %d bytes payload to Python client."), Size);
                 return true; // 성공적으로 크기와 페이로드 전송
            }
            else
            {
                UE_LOG(LogDigitalTwinComm, Error, TEXT("Failed to send payload bytes to Python client. Sent: %d / %d"), BytesSent, Size);
                 Success = false; // 페이로드 전송 실패
            }
        }
        else
        {
            UE_LOG(LogDigitalTwinComm, Error, TEXT("Failed to send size bytes to Python client. Sent: %d / 4"), BytesSent);
             Success = false; // 크기 정보 전송 실패
        }
    }

    // 전송 실패 처리
    if (!Success)
    {
        int32 ErrorCode = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLastErrorCode();
        UE_LOG(LogDigitalTwinComm, Error, TEXT("SendFramedData error %d. Disconnecting client."), ErrorCode);
        // TODO: 게임 스레드에서 안전하게 클라이언트 연결 끊김 처리 신호 발생
        StopServer(); // 간단하게 서버 전체 중지
    }

    return Success; // 전송 성공 시 true, 실패 시 false 반환
}

// Python 클라이언트로 JSON 응답 전송
bool UPythonCommServerComponent::SendJsonResponse(int32 ClientConnectionId, const FString& Status, const FString& Message, const TSharedPtr<FJsonObject>& Data, int32 SequenceId)
{
    if (!ClientSocket || ClientSocket->GetConnectionState() != SCS_Connected)
    {
         UE_LOG(LogDigitalTwinComm, Warning, TEXT("Cannot send JSON response, client socket not connected."));
         return false;
    }
    // NOTE: ClientConnectionId는 현재 단일 클라이언트 모델에서 사용되지 않음.
    // 다중 클라이언트 시 각 연결 소켓을 식별하는 용도로 사용.

    // 응답 JSON 오브젝트 생성
    TSharedPtr<FJsonObject> ResponseObject = MakeShareable(new FJsonObject());
    ResponseObject->SetStringField(TEXT("status"), Status); // "success" 또는 "failure"
    ResponseObject->SetStringField(TEXT("message"), Message); // 상세 메시지
    if (Data.IsValid())
    {
        ResponseObject->SetObjectField(TEXT("data"), Data); // 데이터 포함 (예: get_status, run_simulation 결과)
    }
    // Sequence ID 포함 (Python 클라이언트가 요청-응답 매칭에 사용)
    ResponseObject->SetNumberField(TEXT("sequence_id"), SequenceId);


    // JSON 오브젝트를 FString으로 직렬화
    FString ResponseString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&ResponseString); // JSON Writer 생성
    FJsonSerializer::Serialize(ResponseObject.ToSharedRef(), Writer); // 직렬화

    UE_LOG(LogDigitalTwinComm, Log, TEXT("Sending JSON Response: %s"), *ResponseString);
    // 프레임 형식으로 데이터 전송
    return SendFramedData(ResponseString);
}


// Robot Actor 등록 함수
void UPythonCommServerComponent::RegisterRobotActor(int32 RobotId, AActor* RobotActor)
{
    if (!RobotActor)
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Attempted to register null Robot Actor for ID %d."), RobotId);
        return;
    }
    if (RegisteredRobots.Contains(RobotId))
    {
        UE_LOG(LogDigitalTwinComm, Warning, TEXT("Robot ID %d is already registered (%s). Overwriting with %s."), RobotId, *RegisteredRobots[RobotId]->GetName(), *RobotActor->GetName());
    }
    RegisteredRobots.Add(RobotId, RobotActor);
    UE_LOG(LogDigitalTwinComm, Log, TEXT("Registered Robot Actor '%s' with ID %d."), *RobotActor->GetName(), RobotId);
}