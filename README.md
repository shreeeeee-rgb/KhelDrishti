# KhelDrishti

Edge AI sports biomechanics coach for grassroots athletes.

## Run

```powershell
cd "$env:USERPROFILE\.gemini\antigravity-ide\scratch\kheldrishti"
& .\venv\Scripts\python.exe sample_data\generate_sample_data.py
& .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000

Do not use `--reload` while analysing video: writing HUD clips into `outputs/` can restart the watcher and drop the request.

Webcam live mode needs a browser camera permission grant.

## Tests

```powershell
& .\venv\Scripts\pytest.exe tests/
```
