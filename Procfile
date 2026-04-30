release: python manage.py migrate
web: gunicorn core_api.wsgi --bind 0.0.0.0:$PORT
