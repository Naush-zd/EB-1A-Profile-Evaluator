$ErrorActionPreference = "Stop"
$env:LLM_PROVIDER = "mock"
$demo = Join-Path (Get-Location) ".demo-tmp"
New-Item -ItemType Directory -Force $demo | Out-Null

$source = Join-Path $demo "uscis_demo_source.txt"
$profile = Join-Path $demo "candidate.txt"
$chunks = Join-Path $demo "chunks.jsonl"
$manifest = Join-Path $demo "manifest.json"
$assessment = Join-Path $demo "assessment.json"

@"
EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts.
"@ | Set-Content -Path $source -Encoding utf8
"Received a national award for machine learning excellence." | Set-Content -Path $profile -Encoding utf8

python -m eb1a build-kb $source --url "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2" --out $chunks --manifest $manifest
python -m eb1a inspect-corpus $chunks
python -m eb1a retrieve $chunks "awards prizes recognition" --top-k 3
python -m eb1a assess $profile --corpus $chunks --out $assessment
Write-Host "Assessment written to $assessment"
