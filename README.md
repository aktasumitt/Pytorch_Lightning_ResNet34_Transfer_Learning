# Hayvan Resmi Sınıflandırma Projesi

Bu proje, PyTorch Lightning kullanılarak ResNet34 modelinin transfer öğrenme yöntemiyle eğitildiği bir hayvan resim sınıflandırma modelini içermektedir. Model, yalnızca son FC (Fully Connected) katmanı değiştirilerek 10 farklı hayvan etiketi ile eğitildi. Eğitim süreci 10 epoch sürdü ve %97 doğruluk (accuracy) sağlandı. Ayrıca, MLflow ile modelin takibi yapılmış, DVC (Data Version Control) ile veri sürüm kontrolü gerçekleştirilmiştir. Modelin kullanımı Flask tabanlı bir web uygulaması üzerinden sağlanmaktadır.

**Dataset Link:** <https://www.kaggle.com/datasets/alessiocorrado99/animals10>

## Kullanım

1. **Proje Kurulumu**
   - Projeyi yerel bilgisayarınıza klonlayın:
     ```bash
     git clone https://github.com/aktasumitt/Pytorch_Lightning_ResNet34_Transfer_Learning.git
     cd Pytorch_Lightning_ResNet34_Transfer_Learning
     ```

   - Gerekli bağımlılıkları yüklemek için `requirements.txt` dosyasını kullanın:
     ```bash
     pip install -r requirements.txt
     ```

2. **Modeli İndirme ve Kaydetme**
   - Başlangıç olarak, modeli indirmek ve kaydetmek için aşağıdaki komutu çalıştırın:
     ```bash
     python -m src.pipeline.model_pipeline
     ```

3. **Flask Uygulamasını Çalıştırma**
   - Flask uygulamasını başlatmak için:
     ```bash
     python app.py
     ```

4. **Modeli Kullanma**
   - Flask uygulaması, modelin tahminlerini sağlamak için kullanılabilir. Çalışan uygulama üzerinden modelin tahminlerini alabilirsiniz.

## Model Eğitimi ve Performansı

- **Model**: ResNet34 (PyTorch Lightning)
- **Yöntem**: Transfer öğrenme, son Fully Connected katmanının değiştirilmesi
- **Eğitim Verisi**: 10 farklı hayvan kategorisi
- **Epoch Sayısı**: 10
- **Doğruluk (Accuracy)**: %97

## Teknolojiler

- **PyTorch Lightning**: Model eğitimi ve yönetimi
- **MLflow**: Model takip ve versiyon kontrolü
- **DVC (Data Version Control)**: Veri sürüm kontrolü
- **Flask**: Web uygulaması

## Notlar

- Modelin eğitim süreci ve veri yönetimi için DVC ve MLflow kullanılmıştır.
- Flask uygulaması, kullanıcıların modelden tahmin alabilmesi için bir arayüz sunmaktadır.

