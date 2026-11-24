.PHONY: up down logs test

# Build and start the whole stack: backend :8000, frontend :3000
up:
	docker compose up --build -d
	@echo "Frontend: http://localhost:3000   Backend: http://localhost:8000/health"

down:
	docker compose down

logs:
	docker compose logs -f

# Backend tests, run inside a throwaway container (no local Python setup needed)
test:
	docker run --rm -v "$(CURDIR)/backend:/src" -w /src python:3.12-slim \
		sh -c "pip install -q -r requirements-dev.txt && python -m pytest -q"
