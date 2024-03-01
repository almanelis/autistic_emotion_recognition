# CNN модель для распознавания эмоций людей с РАС
## Как протестировать модель с помощью CV2 и детекцией лица в живом времени
1. Добавляем и активируем виртуальное окружение
```
python -m venv venv
```
```
source venv/Scripts/activate
```
2. Клонируем репозиторий
```
git clone git@github.com:almanelis/autistic_emotion_recognition.git
```
```
cd autistic_emotion_recognition
```
3. Устанавливаем зависимости
```
pip install -r requirements.txt
```
4. Запускаем CV2
```
python run.py
```
## Модели нужно скачать с [Google Drive](https://drive.google.com/drive/folders/1pKhX8fgOk3TT3dIX2zZ3S8bhcrkLt09m?usp=sharing)

## Также можно протестировать CNN модель на фото, записав путь к фото в переменную image_path
```
python photo.py
```
