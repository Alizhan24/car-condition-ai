# 🚗 Car Condition AI — Cleanliness + Damage

Streamlit-приложение для автоматического анализа фотографий автомобиля:
- 🧼 Классификация чистоты кузова (clean / dirty / super dirty и т.д.)
- 🚨 Детекция повреждений (вмятины, царапины, ржавчина)
- 📊 Оценка серьёзности повреждений по площади боксов
- 🖼️ Визуализация с аннотациями и сохранение результатов

Приложение использует **Roboflow Workflows API** для инференса моделей.

---

## 🔧 Установка

1. Склонируй репозиторий:
```bash
git clone https://github.com/Alizhan24/car-condition-ai.git
cd car-condition-ai


python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate


pip install -r requirements.txt


streamlit run app.py


http://localhost:8501


app.py              # Основное приложение Streamlit
requirements.txt    # Список зависимостей
README.md           # Документация проекта
