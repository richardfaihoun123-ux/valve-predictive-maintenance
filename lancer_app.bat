   @echo off
   call .venv\Scripts\activate
   set PYTHONPATH=src
   start http://127.0.0.1:5000
   python -m flask --app app.app run