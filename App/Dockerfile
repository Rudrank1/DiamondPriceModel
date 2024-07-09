FROM python:3.9
RUN pip install Flask
WORKDIR /app
COPY . /app
EXPOSE 5000
ENTRYPOINT ["python", "main.py"]