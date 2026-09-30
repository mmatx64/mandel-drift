param(
  [Parameter(Mandatory=$true)][string]$Executable,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [int]$Seconds = 15,
  [string]$TestMode = '--cycle-test'
)
$ErrorActionPreference = 'Stop'
$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path
$probeProcess = Start-Process -FilePath $executablePath -ArgumentList $TestMode `
  -WorkingDirectory $outputPath -WindowStyle Hidden -PassThru `
  -RedirectStandardError (Join-Path $outputPath 'memory-test-error.txt')
$samples = @()
try {
  for ($sampleIndex = 0; $sampleIndex -lt $Seconds; ++$sampleIndex) {
    Start-Sleep -Seconds 1
    $probeProcess.Refresh()
    if ($probeProcess.HasExited) { break }
    $counter = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -Filter "IDProcess=$($probeProcess.Id)"
    $samples += [pscustomobject]@{
      Sample = $sampleIndex
      WorkingSetMiB = [math]::Round($probeProcess.WorkingSet64 / 1MB, 2)
      PrivateCommitMiB = [math]::Round($probeProcess.PrivateMemorySize64 / 1MB, 2)
      PrivateWorkingSetMiB = if ($null -ne $counter) { [math]::Round($counter.WorkingSetPrivate / 1MB, 2) } else { $null }
      PeakWorkingSetMiB = [math]::Round($probeProcess.PeakWorkingSet64 / 1MB, 2)
    }
  }
} finally {
  if (!$probeProcess.HasExited) {
    $null = $probeProcess.CloseMainWindow()
    if (!$probeProcess.WaitForExit(3000)) { Stop-Process -Id $probeProcess.Id }
  }
  $samples | Export-Csv (Join-Path $outputPath 'process-memory.csv') -NoTypeInformation
  $samples | Format-Table
}
