// Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/DigitalTwinCommPlugin.Build.cs

using UnrealBuildTool;

public class DigitalTwinCommPlugin : ModuleRules
{
	public DigitalTwinCommPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...
			}
			);


		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);


		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Sockets", // 소켓 통신 필요
				"Networking", // 네트워킹 필요
                "Json", // JSON 파싱 필요
                "JsonUtilities" // JSON <-> UObject/UStruct 변환 유틸리티
				// ... add other public dependencies that you statically link with here ...
			}
			);


		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore"
				// ... add private dependencies that you statically link with here ...
			}
			);


		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module desires to load dynamically here ...
			}
			);
	}
}