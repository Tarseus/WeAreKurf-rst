param(
    [Parameter(Mandatory = $true)]
    [string]$LocalPath,

    [string]$RemoteHost = "g51",
    [string]$RemotePath = "/data1/gushengda/deepfake_detection_dfgc/datasets",
    [string]$RemoteName = ""
)

$ErrorActionPreference = "Stop"

$resolved = Resolve-Path -LiteralPath $LocalPath
$item = Get-Item -LiteralPath $resolved
if (-not $RemoteName) {
    $RemoteName = $item.Name
}

Write-Host "Local:  $($item.FullName)"
Write-Host "Remote: ${RemoteHost}:${RemotePath}/${RemoteName}"

ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "mkdir -p '$RemotePath/$RemoteName'"

if ($item.PSIsContainer) {
    Push-Location -LiteralPath $item.FullName
    try {
        tar -cf - . | ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "cd '$RemotePath/$RemoteName' && tar -xf -"
    }
    finally {
        Pop-Location
    }
}
else {
    scp -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 -p "$($item.FullName)" "${RemoteHost}:${RemotePath}/${RemoteName}/"
}

ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "du -sh '$RemotePath/$RemoteName'; find '$RemotePath/$RemoteName' -maxdepth 2 -type f | head"
