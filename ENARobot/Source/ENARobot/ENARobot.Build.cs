// Source/ENARobot/ENARobot.Build.cs

using UnrealBuildTool;

public class ENARobot : ModuleRules
{
	public ENARobot(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "Sockets", // 소켓 통신 필요
            "Networking", // 네트워킹 필요
            "Json", // JSON 파싱 필요
            "JsonUtilities", // JSON <-> UObject/UStruct 변환 유틸리티 (선택적이지만 유용)
            "DigitalTwinCommPlugin" // 우리가 만들 커스텀 플러그인 모듈에 의존성 추가
        });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add steam_api.lib to ThirdParty/Steamworks/Steamv132/lib/Win64
		// and add the following line to OnlineSubsystemSteam.Build.cs
		// PrivateDependencyModuleNames.Add("OnlineSubsystemSteam");
	}
}