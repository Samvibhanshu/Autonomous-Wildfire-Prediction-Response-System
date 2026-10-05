"""python run.py   starts the dashboard on http://127.0.0.1:5000 (set WILDFIRE_API_KEY to protect write endpoints)."""
import os

from wfire.webapp import create_app

if __name__ == "__main__":
    create_app(db_path=os.environ.get("WILDFIRE_DB", "wildfire.db")).run(port=int(os.environ.get("PORT", 5000)))
