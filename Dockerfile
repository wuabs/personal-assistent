FROM python:3.13.1

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание директории приложения
WORKDIR /app

# Копирование файлов
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Порт по умолчанию для Gradio
EXPOSE 7860

# Запуск приложения
CMD ["python", "app.py"]
