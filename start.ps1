<#
  HeadCount — Start Script
  Launches both the FastAPI backend and the Vite dev server.
#>

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HeadCount - AI Classroom Counter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start backend
Write-Host "[1/2] Starting FastAPI backend on port 8000..." -ForegroundColor Green
$backend = Start-Process -FilePath "cmd" -ArgumentList "/c cd backend && uvicorn main:app --reload --port 8000" -PassThru -WindowStyle Normal

Start-Sleep -Seconds 2

# Start frontend
Write-Host "[2/2] Starting Vite dev server..." -ForegroundColor Green
$frontend = Start-Process -FilePath "cmd" -ArgumentList "/c cd frontend && npm run dev" -PassThru -WindowStyle Normal

Write-Host ""
Write-Host "Both servers starting!" -ForegroundColor Yellow
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in the server windows to stop." -ForegroundColor Gray
