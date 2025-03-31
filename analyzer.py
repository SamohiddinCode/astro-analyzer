import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_pdf import PdfPages

IMAGES_DIR = "images"
RESULTS_DIR = "results"
VISUAL_DIR = os.path.join(RESULTS_DIR, "visual")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def analyze_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)  

    if image is None or color_image is None:
        print(f"[!] Не удалось загрузить: {image_path}")
        return []

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        brightness = image[cy, cx]

        cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(color_image, (cx, cy), 3, (0, 0, 255), -1)

        objects.append({
            'filename': os.path.basename(image_path),
            'x': cx,
            'y': cy,
            'brightness': int(brightness),
            'area': float(area)
        })

    out_path = os.path.join(VISUAL_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, color_image)

    return objects

def process_all_images():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    full_paths = [os.path.join(IMAGES_DIR, f) for f in image_files]

    all_objects = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(analyze_image, full_paths)
        for result in results:
            all_objects.extend(result)

    sorted_objects = sorted(all_objects, key=lambda x: -x['brightness'])
    save_to_csv(sorted_objects)
    generate_plots(sorted_objects)


def save_to_csv(data):
    csv_path = os.path.join(RESULTS_DIR, "analysis_results.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'x', 'y', 'brightness', 'area'])
        writer.writeheader()
        for obj in data:
            writer.writerow(obj)
    print(f"[✓] Результаты сохранены в {csv_path}")


def generate_plots(data):
    brightness_values = [obj['brightness'] for obj in data]
    area_values = [obj['area'] for obj in data]

    brightness_path = os.path.join(PLOTS_DIR, "brightness_hist.png")
    area_path = os.path.join(PLOTS_DIR, "area_hist.png")
    pdf_path = os.path.join(PLOTS_DIR, "summary_report.pdf")

    fig1 = plt.figure()
    plt.hist(brightness_values, bins=30, color='blue', alpha=0.7)
    plt.title("Яркость объектов")
    plt.xlabel("Яркость")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.savefig(brightness_path)
    plt.show()

    fig2 = plt.figure()
    plt.hist(area_values, bins=30, color='green', alpha=0.7)
    plt.title("Площади объектов")
    plt.xlabel("Площадь")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.savefig(area_path)
    plt.show()

    #  в PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
    print(f"[✓] Графики сохранены в {PLOTS_DIR} и PDF-отчёт создан")

if __name__ == "__main__":
    process_all_images()
