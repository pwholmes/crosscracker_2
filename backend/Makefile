run:
	: > server.log && ../.venv/bin/uvicorn server:app --app-dir src --port 8000 --reload 2>&1 | tee -a server.log
test:
	../.venv/bin/pytest
itest:
	../.venv/bin/pytest -m integration
