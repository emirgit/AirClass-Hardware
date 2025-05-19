# AirClass Hand Gesture Pipeline Kullanım Rehberi

Bu projede el hareketlerinden veri çıkarımı, CSV birleştirme ve makine öğrenmesi modeli eğitimi için üç ana Python dosyası kullanılır. Aşağıda her bir dosyanın **adım adım nasıl kullanılacağı**, hangi dosya/dizin adlarının girileceği ve kullanıcıya sunulan seçenekler detaylı şekilde açıklanmıştır.

---

## 1. extract_landmarks.py

**Amaç:**  
Belirttiğiniz ana klasörün altındaki gesture klasörlerindeki tüm resimlerden MediaPipe ile el landmark'larını çıkarır ve her gesture için ayrı bir CSV dosyasına kaydeder.

### Kullanım

```bash
python extract_landmarks.py
```

1. **Klasör ismi girme:**

   Program başında, resimlerin bulunduğu ana klasörün adını girmeniz istenir.
   Örnek:
   ```shell
   Enter the name of the folder that contains the images: (like: hagrid-512p)
   > hagrid-512p
   ```
   Bu klasörün içinde her bir gesture için ayrı alt klasörler (ör: call, ok, stop vs.) olmalı.

2. **Gesture seçimi:**

   Program, klasördeki tüm gesture klasörlerini listeler ve size şöyle bir seçenek sunar:
   ```shell
   Select gestures to process (enter numbers separated by space, or 'all' for all):
   > all
   ```
   `all` yazarsanız tüm gesture'lar işlenir.
   Veya sadece istediğiniz gesture'ları numara ile seçebilirsiniz (ör: 1 3 5).

3. **Devam onayı:**

   Seçiminizi yaptıktan sonra:
   ```shell
   Press 'y' to proceed or any other key to exit:
   > y
   ```
   `y` tuşuna basarsanız landmark çıkarımı başlar, başka bir tuşa basarsanız program sonlanır.

4. **Çıktı:**

   Otomatik olarak her gesture için en fazla 2000 resim işlenir ve landmark'lar `all_landmarks/` klasörüne CSV olarak kaydedilir.

---

## 2. combine_landmarks.py

**Amaç:**  
Birden fazla gesture CSV dosyasını birleştirerek tek bir CSV dosyası oluşturur. Her gesture için maksimum örnek sayısı belirleyebilirsiniz.

### Kullanım

```bash
python combine_landmarks.py
```

1. **Klasör seçimi:**
   ```
   Output directory (default: all_landmarks):
   ```
   Varsayılan olarak `all_landmarks` klasörünü kullanır. İsterseniz başka bir klasör da belirtebilirsiniz.

2. **CSV seçimi:**
   ```shell
   Select CSV files to combine (enter numbers separated by space, or 'all' for all):
   > all
   ```
   Birleştirmek istediğiniz CSV dosyalarını seçin.

3. **Limit belirleme:**
   Her dosya için maksimum kaç örnek alınacağını belirleyebilirsiniz. Tüm dosyalar için aynı limiti veya her biri için ayrı limit girebilirsiniz.

4. **Çıktı:**
   Seçilen CSV dosyaları birleştirilir ve `all_landmarks_combined.csv` olarak belirtilen klasöre kaydedilir.

---

## 3. train_model.py

**Amaç:**  
Birleştirilmiş landmark CSV dosyasını kullanarak bir makine öğrenmesi modeli (Keras ile) eğitir ve modeli kaydeder.

### Kullanım

```bash
python train_model.py
```

1. **Model dosya adı:**
   ```
   Model file name to save (default: gesture_recognizer.keras):
   ```
   Modelin kaydedileceği dosya adını girin veya Enter ile varsayılanı kullanın.

2. **Onay:**
   ```
   Proceed with this model file? (y/n):
   ```
   `y` ile devam edin.

3. **Eğitim:**
   Model eğitimi başlar. Eğitim sırasında doğruluk ve kayıp değerleri ekrana yazılır.

4. **Çıktılar:**
   - Eğitilmiş model (`.keras` uzantılı)
   - Label encoder (`label_encoder.pkl`)
   - Scaler (`scaler.pkl`)
   - Eğitim geçmişi grafiği (`training_history.png`)

---

## Notlar

- Tüm adımlar sırasıyla uygulanmalıdır: Önce landmark çıkarımı, sonra CSV birleştirme, en son model eğitimi.
- Her adımda çıktı dosyalarının doğru klasörlerde oluştuğundan emin olun.
- Model eğitimi için `tensorflow`, `scikit-learn`, `pandas`, `matplotlib` gibi kütüphaneler gereklidir.

---