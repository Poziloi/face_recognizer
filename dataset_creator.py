# dataset_creator.py

import os
import cv2
import sys

def create_dataset_for_user(user_id, user_name, num_samples=200, save_dir="faces"):
    # Инициализируем веб-камеру
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Создаем папку для нового пользователя в директории faces
    save_dir = os.path.abspath(save_dir)  # Используем абсолютный путь
    user_folder = os.path.join(save_dir, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            # Сохраняем изображения в папку пользователя
            image_path = os.path.join(user_folder, f"user_{user_id}_{count}.jpg")
            cv2.imwrite(image_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"Samples: {count}/{num_samples}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Collecting faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python dataset_creator.py <user_id> <user_name>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    user_name = sys.argv[2]

    create_dataset_for_user(user_id, user_name)