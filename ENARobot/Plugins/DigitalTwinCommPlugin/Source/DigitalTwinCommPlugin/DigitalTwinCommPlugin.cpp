// Plugins/DigitalTwinCommPlugin/Source/DigitalTwinCommPlugin/DigitalTwinCommPlugin.cpp

#include "DigitalTwinCommPlugin.h"
#include "Interfaces/IPluginManager.h" // 플러그인 관리자 사용

#define LOCTEXT_NAMESPACE "FDigitalTwinCommPluginModule"

void FDigitalTwinCommPluginModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file
	UE_LOG(LogTemp, Log, TEXT("DigitalTwinCommPlugin module starting up."));

	// Get the base directory of this plugin
	// FString BaseDir = IPluginManager::Get().FindPlugin("DigitalTwinCommPlugin")->GetBaseDir();

	// Add custom code here to extend the editor or game
}

void FDigitalTwinCommPluginModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that supportunload, call ShutdownModule() to clean up your module.
	UE_LOG(LogTemp, Log, TEXT("DigitalTwinCommPlugin module shutting down."));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FDigitalTwinCommPluginModule, DigitalTwinCommPlugin)