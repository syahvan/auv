import cv2
import numpy as np
import math

def calculate_angle_between_lines(x):
    # Ukuran gambar (sekarang menggunakan frame_width dan frame_height)
    frame_width, frame_height = 480, 480

    # Garis pertama
    p1 = (frame_width // 2, 0)
    p2 = (frame_width // 2, frame_height)

    # Garis kedua
    q1 = (x, 0)
    q2 =(frame_width // 2, frame_height)

    # Menggambar garis pada gambar blank
    image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.line(image, p1, p2, (255, 0, 0), 2)
    cv2.line(image, q1, q2, (0, 255, 0), 2)

    # Menghitung vektor normal untuk masing-masing garis
    vec1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    vec2 = np.array([q2[0] - q1[0], q2[1] - q1[1]])

    # Hitung sudut antara dua vektor
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    # Tentukan apakah sudut positif atau negatif
    if x < frame_width // 2:
        angle_degrees *= -1

    # Tampilkan gambar dengan garis-garis
    cv2.imshow("Garis-garis", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return angle_degrees

# Contoh penggunaan fungsi
x_input = 0  # Nilai x acak
angle = calculate_angle_between_lines(x_input)
print(f"Sudut antara kedua garis untuk x = {x_input}: {angle:.2f} derajat")
