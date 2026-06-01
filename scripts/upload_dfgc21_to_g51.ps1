param(
    [string]$LocalDir = "datasets\DFGC-21",
    [string]$RemoteHost = "g51",
    [string]$RemoteDir = "/data1/gushengda/deepfake_detection_dfgc/datasets/DFGC-21"
)

$ErrorActionPreference = "Stop"

$local = Resolve-Path -LiteralPath $LocalDir
$files = Get-ChildItem -LiteralPath $local -File | Sort-Object Name

ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "mkdir -p '$RemoteDir'"

foreach ($file in $files) {
    $name = $file.Name
    $size = [string]$file.Length
    $remoteSize = ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "if [ -f '$RemoteDir/$name' ]; then stat -c %s '$RemoteDir/$name'; else echo MISSING; fi"
    $remoteSize = ($remoteSize | Select-Object -First 1).Trim()

    if ($remoteSize -eq $size) {
        Write-Host "SKIP $name ($size bytes)"
        continue
    }

    Write-Host "UPLOAD $name local=$size remote=$remoteSize"
    $tmpName = ".$name.uploading"
    ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "rm -f '$RemoteDir/$tmpName'"
    scp -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 -p "$($file.FullName)" "${RemoteHost}:${RemoteDir}/${tmpName}"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for $name"
    }

    $tmpSize = ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "stat -c %s '$RemoteDir/$tmpName'"
    $tmpSize = ($tmpSize | Select-Object -First 1).Trim()
    if ($tmpSize -ne $size) {
        throw "size mismatch for $name after upload: local=$size remote_tmp=$tmpSize"
    }

    ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "mv -f '$RemoteDir/$tmpName' '$RemoteDir/$name'"
    Write-Host "DONE $name"
}

ssh -oBatchMode=yes -oConnectTimeout=8 -oConnectionAttempts=1 $RemoteHost "find '$RemoteDir' -maxdepth 1 -type f -printf '%f %s\n' | sort; du -sh '$RemoteDir'"
