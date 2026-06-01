param(
    [Parameter(Mandatory = $true)]
    [string]$UrlFile,

    [string]$OutputDir = "datasets\downloads"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$urls = Get-Content -LiteralPath $UrlFile | ForEach-Object { $_.Trim() } | Where-Object {
    $_ -and -not $_.StartsWith("#")
}

if ($urls.Count -eq 0) {
    throw "No URLs found in $UrlFile"
}

foreach ($url in $urls) {
    $name = [System.IO.Path]::GetFileName(([Uri]$url).AbsolutePath)
    if (-not $name) {
        $name = "download_$([Math]::Abs($url.GetHashCode())).bin"
    }
    $out = Join-Path $OutputDir $name
    if (Test-Path -LiteralPath $out) {
        Write-Host "Skip existing $out"
        continue
    }
    Write-Host "Downloading $url"
    Write-Host "       -> $out"
    Invoke-WebRequest -Uri $url -OutFile $out
}

Get-ChildItem -LiteralPath $OutputDir | Select-Object Name, Length, LastWriteTime
