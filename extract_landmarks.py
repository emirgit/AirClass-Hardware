import cv2 as cv
import mediapipe as mp
import csv
import os
import glob
import time
import pandas as pd

def get_gesture_folders(data_dir):
    """
    Veri dizinindeki tüm alt klasörleri döndürür (her biri bir gesture olmalı).
    """
    if not os.path.exists(data_dir):
        return []
    
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

def extract_landmarks(gesture_names, data_dir, max_examples, output_dir):
    """
    Seçilen gesture'lar için landmark çıkarma işlemi yapar.
    """
    stats = {}
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    # Her bir gesture için işlem yap
    for i, gesture_name in enumerate(gesture_names):
        examples_collected = 0
        hand_not_detected = 0
        image_error = 0
        processed_images = 0
        
        data_path = os.path.join(data_dir, gesture_name)
        csv_path = os.path.join(output_dir, f"{gesture_name}.csv")
        
        # Klasör var mı kontrol et
        if not os.path.exists(data_path):
            print(f"Skipping {gesture_name}: Directory {data_path} does not exist.")
            stats[gesture_name] = {
                'success': 0, 'hand_not_detected': 0, 'image_error': 0, 'processed': 0, 'success_rate': 0
            }
            continue
            
        print(f"\nProcessing gesture: {gesture_name}")
        
        # Create CSV file and write header
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Create header - 21 landmarks with x, y, z coordinates
            header = []
            for j in range(21):
                header.extend([f'x{j}', f'y{j}', f'z{j}'])
            header.append("label")
            csv_writer.writerow(header)
            
            # Tüm görüntü dosyalarını bir listeye al
            image_files = []
            for format in ['.jpg', '.jpeg', '.png']:
                image_files.extend(glob.glob(os.path.join(data_path, f'*{format}')))
                image_files.extend(glob.glob(os.path.join(data_path, f'*{format.upper()}')))
            
            print(f"Found {len(image_files)} image files for {gesture_name}")
            print(f"Processing up to {max_examples} images...")
            
            # Kaç dosya işleneceğini belirle
            files_to_process = min(len(image_files), max_examples)
            
            # Process image files
            start_time = time.time()
            last_update_time = start_time
            
            for idx, img_path in enumerate(image_files[:files_to_process]):
                processed_images += 1
                
                # İlerleme durumunu göster (her 5 saniyede bir veya her 10 işlemde bir)
                current_time = time.time()
                progress = (idx + 1) / files_to_process * 100
                
                if idx % 10 == 0 or current_time - last_update_time > 5:
                    eta = (current_time - start_time) / (idx + 1) * (files_to_process - idx - 1) if idx > 0 else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                    print(f"\rProgress: {progress:.1f}% | {idx+1}/{files_to_process} images | ETA: {eta_str}", end="")
                    last_update_time = current_time
                
                try:
                    # Read image
                    image = cv.imread(img_path)
                    if image is None:
                        image_error += 1
                        continue
                        
                    # Convert to RGB for MediaPipe
                    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    
                    # Process image with MediaPipe
                    results = hands.process(image_rgb)
        
                    if results.multi_hand_landmarks:
                        # Get the first detected hand
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        row_data = []
                        
                        # Extract and normalize landmarks
                        for landmark in hand_landmarks.landmark:
                            # Add x, y, z coordinates
                            row_data.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Add gesture label at the end
                        row_data.append(gesture_name)
                        
                        csv_writer.writerow(row_data)
                        examples_collected += 1
                    else:
                        hand_not_detected += 1
        
                except Exception as e:
                    print(f"\nError processing {img_path}: {str(e)}")
                    image_error += 1
            
            # Son ilerleme durumu
            print(f"\rProgress: 100% | {files_to_process}/{files_to_process} images | Complete")
            
            success_rate = (examples_collected / processed_images * 100) if processed_images > 0 else 0
            
            print(f"Total examples collected for {gesture_name}: {examples_collected}")
            print(f"Hands not detected: {hand_not_detected}")
            print(f"Image errors: {image_error}")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Data saved to {csv_path}")
            
            # İstatistikleri kaydet
            stats[gesture_name] = {
                'success': examples_collected,
                'hand_not_detected': hand_not_detected,
                'image_error': image_error,
                'processed': processed_images,
                'success_rate': success_rate
            }
    
    # Generate report
    print("\n" + "="*60)
    print("LANDMARK EXTRACTION REPORT")
    print("="*60)
    print(f"{'Gesture':<15} {'Success':<10} {'Not Detected':<15} {'Error':<10} {'Success Rate':<15}")
    print("-"*60)
    
    total_success = 0
    total_not_detected = 0
    total_error = 0
    total_processed = 0
    
    for gesture, data in stats.items():
        print(f"{gesture:<15} {data['success']:<10} {data['hand_not_detected']:<15} {data['image_error']:<10} {data['success_rate']:.2f}%")
        total_success += data['success']
        total_not_detected += data['hand_not_detected']
        total_error += data['image_error']
        total_processed += data['processed']
    
    overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
    
    print("-"*60)
    print(f"{'TOTAL':<15} {total_success:<10} {total_not_detected:<15} {total_error:<10} {overall_success_rate:.2f}%")
    print("="*60)
    
    # Clean up
    hands.close()

def main():
    # Output directory
    output_dir = "all_landmarks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Kullanıcıdan veri klasörü ismini al
    data_dir = input("Enter the name of the folder that contains the images: (like: hagrid-512p)").strip()
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    gesture_folders = get_gesture_folders(data_dir)
    if not gesture_folders:
        print(f"Error: No gesture folders found in '{data_dir}'.")
        return

    print(f"Data directory: {data_dir}")
    print("Available gestures:")
    for i, folder in enumerate(gesture_folders):
        print(f"{i+1}. {folder}")

    # Kullanıcıdan gesture seçimini al
    print("\nSelect gestures to process (enter numbers separated by space, or 'all' for all):")
    selection = input("> ")

    if selection.lower() == 'all':
        selected_gestures = gesture_folders
    else:
        try:
            indices = [int(x) - 1 for x in selection.split()]
            selected_gestures = [gesture_folders[i] for i in indices if 0 <= i < len(gesture_folders)]
        except ValueError:
            print("Invalid input. Please enter numbers separated by space.")
            return

    if not selected_gestures:
        print("No gestures selected. Exiting.")
        return

    print("\nSelected gestures:", ", ".join(selected_gestures))

    # Maksimum örnek sayısı sabit: 2000
    max_examples = 2000
    print(f"\nWill process up to {max_examples} images for each of the {len(selected_gestures)} selected gestures.")
    is_proceed = input("Press 'y' to proceed or any other key to exit: ")
    
    if is_proceed != "y":
        return

    # İşlemi başlat
    print("\nStarting landmark extraction...")
    extract_landmarks(selected_gestures, data_dir, max_examples, output_dir)
    print("\nLandmark extraction completed.")

if __name__ == '__main__':
    main()
