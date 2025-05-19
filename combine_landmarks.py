import os
import pandas as pd

def combine_all_landmarks():
    # Kullanıcıdan output klasörü al
    default_dir = "all_landmarks"
    output_dir = input(f"Output directory (default: {default_dir}): ").strip()
    if not output_dir:
        output_dir = default_dir

    combined_csv_name = "all_landmarks_combined.csv"

    # Mevcut CSV dosyalarını bul
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' does not exist.")
        return

    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and f != combined_csv_name]
    
    if not csv_files:
        print(f"Error: No CSV files found in {output_dir} directory.")
        return
        
    # Dosyaları listele ve kullanıcıya seçenek sun
    print("\n" + "="*60)
    print("AVAILABLE CSV FILES:")
    print("="*60)
    for i, csv_file in enumerate(csv_files):
        # Dosya adı ve satır sayısını göster
        file_path = os.path.join(output_dir, csv_file)
        try:
            num_rows = sum(1 for _ in open(file_path)) - 1  # header satırını çıkar
            print(f"{i+1}. {csv_file} ({num_rows} landmarks)")
        except:
            print(f"{i+1}. {csv_file} (could not read)")
    
    # Kullanıcıdan CSV seçimini al
    print("\nSelect CSV files to combine (enter numbers separated by space, or 'all' for all):")
    selection = input("> ")
    
    if selection.lower() == 'all':
        selected_files = csv_files
    else:
        try:
            indices = [int(x) - 1 for x in selection.split()]
            selected_files = [csv_files[i] for i in indices if 0 <= i < len(csv_files)]
        except ValueError:
            print("Invalid input. Please enter numbers separated by space.")
            return
    
    if not selected_files:
        print("No files selected. Operation cancelled.")
        return
    
    print("\nSelected files:", ", ".join(selected_files))
    
    # Her dosyadan kaç landmark alınacağını sor
    max_landmarks = {}
    
    # Önce her dosya için ayrı ayrı sor veya hepsi için aynısını sor
    print("\nDo you want to set the same limit for all files or individual limits?")
    print("1. Same limit for all files")
    print("2. Set individual limits")
    limit_option = input("Select option (1/2): ").strip()
    
    if limit_option == '1':
        # Tüm dosyalar için aynı limit
        while True:
            try:
                all_limit = input("\nMax landmarks per file (enter for no limit): ").strip()
                if not all_limit:  # Boş giriş = limit yok
                    all_limit = None
                else:
                    all_limit = int(all_limit)
                    if all_limit <= 0:
                        print("Please enter a positive number or leave empty for no limit.")
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Her dosya için aynı limiti ayarla
        for file in selected_files:
            max_landmarks[file] = all_limit
    
    elif limit_option == '2':
        # Her dosya için ayrı limit
        for file in selected_files:
            while True:
                try:
                    limit = input(f"\nMax landmarks for {file} (enter for no limit): ").strip()
                    if not limit:  # Boş giriş = limit yok
                        max_landmarks[file] = None
                        break
                    
                    limit = int(limit)
                    if limit <= 0:
                        print("Please enter a positive number or leave empty for no limit.")
                        continue
                    
                    max_landmarks[file] = limit
                    break
                except ValueError:
                    print("Please enter a valid number.")
    else:
        print("Invalid option. Using no limits.")
        for file in selected_files:
            max_landmarks[file] = None
    
    # Dosyaları birleştir
    combined_df = pd.DataFrame()
    total_landmarks = 0
    
    print("\nCombining files...")
    for i, csv_file in enumerate(selected_files):
        file_path = os.path.join(output_dir, csv_file)
        try:
            # Dosyayı oku
            df = pd.read_csv(file_path)
            
            # Satır sayısını limit ile kısıtla
            if max_landmarks[csv_file] is not None and len(df) > max_landmarks[csv_file]:
                # Rastgele satırları seç (shuffle ve ilk N satırı al)
                df = df.sample(n=max_landmarks[csv_file], random_state=42).reset_index(drop=True)
            
            # DataFrame'e ekle
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            # İstatistikler
            landmarks_added = len(df)
            total_landmarks += landmarks_added
            print(f"Added {landmarks_added} landmarks from {csv_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    # Son dosyayı kaydet
    combined_csv_path = os.path.join(output_dir, combined_csv_name)
    combined_df.to_csv(combined_csv_path, index=False)
    
    print("\n" + "="*60)
    print(f"COMBINATION COMPLETE: {total_landmarks} total landmarks")
    print(f"Output file: {combined_csv_path}")
    print("="*60)
    
combine_all_landmarks()