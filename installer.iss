; RFlect Inno Setup Script
; Builds a Windows installer from the PyInstaller-generated exe
;
; Usage:
;   iscc installer.iss                          (uses default version)
;   iscc /DRFLECT_VERSION=4.0.0 installer.iss   (override version)

#define MyAppName "RFlect"
#ifndef RFLECT_VERSION
  #define RFLECT_VERSION "4.0.0"
#endif
#define MyAppVersion RFLECT_VERSION
#define MyAppPublisher "RFingAdam"
#define MyAppURL "https://github.com/RFingAdam/RFlect"
#define MyAppExeName "RFlect.exe"

[Setup]
; AppId must stay the same across versions for upgrade detection
AppId={{E8A3F2B1-9C4D-4E5F-8A6B-7D2C1E3F4A5B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes
InfoAfterFile=RELEASE_NOTES.md
SetupIconFile=smith_logo.ico
UninstallDisplayIcon={app}\smith_logo.ico
OutputDir=installer_output
OutputBaseFilename=RFlect_Installer_{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Upgrade support
UsePreviousAppDir=yes
UpdateUninstallLogAppName=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\RFlect.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "smith_logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\smith_logo.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "settings.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "RELEASE_NOTES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\smith_logo.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\smith_logo.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; Clean up old versioned executables from pre-v4 installs
Type: files; Name: "{app}\RFlect_v*.exe"

[UninstallDelete]
Type: files; Name: "{app}\settings.json"
Type: filesandordirs; Name: "{app}\assets"
