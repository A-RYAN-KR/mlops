.PHONY: up down status clean feast-apply feast-materialize train

up:
	docker compose up -d

down:
	docker compose down

status:
	docker compose ps

clean:
	docker compose down -v

feast-apply:
	cd feature_store && ..\.venv\Scripts\feast apply

feast-materialize:
	cd feature_store && ..\.venv\Scripts\feast materialize 2026-08-01T00:00:00 2026-08-15T00:00:00

train:
	.venv\Scripts\python src/models/train.py
