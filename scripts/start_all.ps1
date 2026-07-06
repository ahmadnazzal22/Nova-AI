# Start all RAG Platform services locally
Write-Host "Starting RAG Platform services..." -ForegroundColor Green

# 1. Start Redis (if available)
$redis = Get-Process -Name "redis-server" -ErrorAction SilentlyContinue
if (-not $redis) {
    Write-Host "Starting Redis..." -ForegroundColor Yellow
    Start-Process -WindowStyle Hidden -FilePath "redis-server"
} else {
    Write-Host "Redis already running" -ForegroundColor Green
}

# 2. Start Qdrant (if available)
$qdrant = Get-Process -Name "qdrant" -ErrorAction SilentlyContinue
if (-not $qdrant) {
    Write-Host "Starting Qdrant..." -ForegroundColor Yellow
    Start-Process -WindowStyle Hidden -FilePath "qdrant"
} else {
    Write-Host "Qdrant already running" -ForegroundColor Green
}

# 3. Start API Gateway
Write-Host "Starting API Gateway on port 8000..." -ForegroundColor Cyan
$api = Start-Process -WindowStyle Hidden -PassThru -FilePath "python" -ArgumentList "run_gateway.py"

# 4. Start Celery worker (optional)
Write-Host "Starting Celery worker..." -ForegroundColor Cyan
$worker = Start-Process -WindowStyle Hidden -PassThru -FilePath "python" -ArgumentList "run_celery_worker.py"

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host "  API Gateway: http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:    http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend:    http://localhost:8002" -ForegroundColor White
Write-Host ""
Write-Host "To stop: docker-compose down (if using Docker) or kill the processes manually." -ForegroundColor Yellow
