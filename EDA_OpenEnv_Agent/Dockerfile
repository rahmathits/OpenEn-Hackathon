FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]