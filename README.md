# Baltimore Incentives Matching Tool

## Setup in VS Code

1. Open this folder in VS Code (`File > Open Folder`)
2. Open a terminal in VS Code (`Terminal > New Terminal`)
3. Create a virtual environment:
   ```
   python3 -m venv venv
   ```
4. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
5. Install VS Code's Python extension if you don't have it (search "Python" in Extensions, the Microsoft one)
6. Point VS Code at the venv: Cmd/Ctrl+Shift+P -> "Python: Select Interpreter" -> choose the one inside `./venv`
7. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
8. Run the app:
   ```
   uvicorn app.main:app --reload
   ```
9. Open http://127.0.0.1:8000 in your browser — you should see `{"status": "running", ...}`
10. Try http://127.0.0.1:8000/test-filter — this runs a sample company through the rules engine so you can see it work end to end.

## Project structure so far

```
incentives_app/
├── app/
│   ├── main.py           <- FastAPI entry point
│   ├── data_loader.py    <- loads the 3 datasets into memory
│   └── rules_engine.py   <- stage 1 hard filter (before Gemini)
├── data/
│   ├── Incentives_Classifier_Normalized.xlsx
│   ├── Known_Companies_Clean.xlsx
│   └── Venture_Rounds_Clean.xlsx
├── templates/            <- (empty, home screen UI goes here next)
├── static/               <- (empty, CSS/JS goes here next)
└── requirements.txt
```

## Not built yet
- Gemini ranking layer (stage 2 of the matching pipeline)
- Home screen / question-flow UI
- Deployment config
