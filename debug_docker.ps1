Write-Host "=== Docker Compose Status ===" -ForegroundColor Cyan
docker-compose ps

Write-Host "`n=== Docker Container Logs ===" -ForegroundColor Cyan
docker-compose logs

Write-Host "`n=== Container File Structure ===" -ForegroundColor Cyan
docker-compose exec paper-extractor ls -la /app
docker-compose exec paper-extractor ls -la /app/model-dir

Write-Host "`n=== Python Package Versions ===" -ForegroundColor Cyan
docker-compose exec paper-extractor pip list | findstr "torch transformers numpy safetensors"

Write-Host "`n=== System Resources ===" -ForegroundColor Cyan
docker-compose exec paper-extractor nvidia-smi
if ($LASTEXITCODE -ne 0) { Write-Host "nvidia-smi not available" }