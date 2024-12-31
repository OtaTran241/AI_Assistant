FROM python:3.10

WORKDIR /AI_Assistant

COPY . /AI_Assistant

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/Main.py"]
