FROM python:3.9
RUN pip install requests beautifulsoup4
WORKDIR /app
COPY . /app
ENTRYPOINT ["python", "main.py"]