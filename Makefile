export PYTHONPATH := .
install:   ; pip install -r requirements.txt -r requirements-ui.txt -r requirements-dev.txt
test:      ; pytest -q
lint:      ; ruff check .
train:     ; python scripts/bootstrap_demo.py
api:       ; AUTO_BOOTSTRAP=true uvicorn backend.api.main:app --reload --port 8000
ui:        ; streamlit run frontend/app.py
up:        ; docker compose up --build
k8s-apply: ; kubectl apply -f deployment/kubernetes/
