FROM python:3.10-slim

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["gradio", "gui.py"]