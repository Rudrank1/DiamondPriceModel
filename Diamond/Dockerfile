# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the code into the container
COPY . .

# Install the required packages
RUN pip install -r requirements.txt

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Set Streamlit configuration options to ensure it listens on the correct IP address
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0

# Define the command to run the app
CMD ["streamlit", "run", "app.py"]
