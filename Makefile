.PHONY: make_requirements

make_requirements:
	poetry export -f requirements.txt -o requirements.txt

init_db:
	sqlite3 data/surv.db "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT);"
