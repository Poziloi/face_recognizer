# face_recognition.py

import cv2
import numpy as np
import os
import datetime
import json
import argparse

USERS_FILE = "users.json"
TRAINED_FILES = {"LBPH": "trained_faces_lbph.yml", "Fisherfaces": "trained_faces_fisher.yml"}

def load_users():
    """
    Считываем users.json и конвертируем ключи (ID) из str в int,
    чтобы они совпадали с типом predicted_id, который возвращает recognizer.predict().
    """
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Преобразуем ключи словаря к int, а значения (имена) оставляем нетронутыми
    user_dict = {}
    for k, v in data.items():
        try:
            user_dict[int(k)] = v
        except ValueError:
            # если почему-то ключ не число, можно пропустить или обработать иначе
            pass
    return user_dict

# Загружаем словарь имен (ID -> Имя)
USER_NAMES = load_users()

def load_recognizer(algorithm):
    """
    Загружаем алгоритм распознования на основе выбора (LBPH или Fisherfaces)
    и подгружаем обученную модель (обученная модель создается после обучения (tariner.py))
    """
    if algorithm == "LBPH":
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        model_file = TRAINED_FILES["LBPH"]
    elif algorithm == "Fisherfaces":
        recognizer = cv2.face.FisherFaceRecognizer_create()
        model_file = TRAINED_FILES["Fisherfaces"]
    else:
        raise ValueError("Неизвестный алгоритм: выберите 'LBPH' или 'Fisherfaces'.")

    if os.path.exists(model_file):
        recognizer.read(model_file) # Загружаем данные в алгоритм из файла
        return recognizer
    else:
        print(f"Ошибка: не найден файл модели '{model_file}'. Сначала запустите тренеровку (trainer.py).")
        return None

def recognize_from_image(image_path, algorithm):
    screen_width = 800  # Максимальная ширина окна
    screen_height = 600  # Максимальная высота окна

    recognizer = load_recognizer(algorithm)
    if recognizer is None:
        return

    if not os.path.exists(image_path):
        print(f"Ошибка: файл изображения '{image_path}' не найден.")
        return

    frame = cv2.imread(image_path)
    # Получаем размеры изображения
    img_height, img_width = frame.shape[:2]

    # Рассчитываем коэффициент масштабирования
    scale_width = screen_width / img_width
    scale_height = screen_height / img_height
    scale = min(scale_width, scale_height, 1)  # Берем минимальный коэффициент

    if frame is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'.")
        return
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        try:
            resized_roi_gray = cv2.resize(roi_gray, (200, 200))
            predicted_id, confidence = recognizer.predict(resized_roi_gray)
        except cv2.error as e:
            print(f"Ошибка при распознавании лица: {e}")
            continue

        # Преобразуем confidence в некий "процент уверенности"
        # (чем меньше confidence у LBPH, тем выше реальная уверенность)
        if algorithm == "LBPH":
            confidence_text = max(0, min(100, int(100 - confidence)))
        else:
            confidence_text = max(0, min(100, (int(1000 - confidence)/10)))

        if predicted_id in USER_NAMES and confidence_text > 15:
            name = USER_NAMES[predicted_id]
            label = f"{name} ({confidence_text}%)"
            color = (0, 255, 0)
        else:
            label = f"Неопознанный ({confidence_text}%)"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    
    # Применяем масштабирование
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Face Recognition - Image", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video_stream(video_source, delay, algorithm):
    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Инициализируем распознаватель и подгружаем обученную модель
    recognizer = load_recognizer(algorithm)
    if recognizer is None:
        return

    # Открываем видеопоток
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видеопоток {video_source}. Проверьте его доступность и права.")
        return

    # Файл для логов (открываем в режиме добавления)
    log_file = open("face_log.txt", "a", encoding="utf-8")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с видеопотока.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Вырезаем лицо из кадра
            roi_gray = gray[y:y+h, x:x+w]

            # Пытаемся распознать
            try:
                resized_roi_gray = cv2.resize(roi_gray, (200, 200))
                predicted_id, confidence = recognizer.predict(resized_roi_gray)
            except cv2.error as e:
                print(f"Ошибка при распознавании лица: {e}")
                continue

            # Преобразуем confidence в некий "процент уверенности"
            # (чем меньше confidence у LBPH, тем выше реальная уверенность)
            if algorithm == "LBPH":
                confidence_text = max(0, min(100, int(100 - confidence)))
            else:
                confidence_text = max(0, min(100, (int(1000 - confidence)/10)))

            # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
            if predicted_id in USER_NAMES and confidence_text > 15:
                name = USER_NAMES[predicted_id]
                label = f"{name} ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Распознан: {name}, Уверенность: {confidence_text}%\n")
                color = (0, 255, 0)
            else:
                label = f"Неопознанный ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                color = (0, 0, 255)

            # Рисуем рамку и подпись на кадре
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Закрываем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--stream", type=str, help="URL of the video stream")
    parser.add_argument("--delay", type=int, default=30, help="Delay between frames in milliseconds")
    parser.add_argument("--image", type=str, help="Путь к изображению для распознавания")
    parser.add_argument("--algorithm", type=str, choices=["LBPH", "Fisherfaces"], default="LBPH", help="Выбор алгоритма распознавания")
    args = parser.parse_args()

    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Инициализируем распознаватель и подгружаем обученную модель
    recognizer = load_recognizer(args.algorithm)
    if recognizer is None:
        return

    if args.image:
        recognize_from_image(args.image, args.algorithm)
    elif args.stream:
        process_video_stream(args.stream, args.delay, args.algorithm)
    else:
        # Открываем веб-камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру. Проверьте её доступность и права.")
            return

        # Файл для логов (открываем в режиме добавления)
        log_file = open("face_log.txt", "a", encoding="utf-8")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Вырезаем лицо из кадра
                roi_gray = gray[y:y+h, x:x+w]

                # Пытаемся распознать
                try:
                    resized_roi_gray = cv2.resize(roi_gray, (200, 200))
                    predicted_id, confidence = recognizer.predict(resized_roi_gray)
                except cv2.error as e:
                    print(f"Ошибка при распознавании лица: {e}")
                    continue

                # Преобразуем confidence в некий "процент уверенности"
                # (чем меньше confidence у LBPH, тем выше реальная уверенность)
                if args.algorithm == "LBPH":
                    confidence_text = max(0, min(100, int(100 - confidence)))
                else:
                    confidence_text = max(0, min(100, (int(1000 - confidence)/10)))

                # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
                if predicted_id in USER_NAMES and confidence_text > 15:
                    name = USER_NAMES[predicted_id]
                    label = f"{name} ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Распознан: {name}, Уверенность: {confidence_text}%\n")
                    color = (0, 255, 0)
                else:
                    label = f"Неопознанный ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                    color = (0, 0, 255)

                # Рисуем рамку и подпись на кадре
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)

            # Нажмите 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Закрываем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

if __name__ == "__main__":
    main()