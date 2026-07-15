# YOLO Mimari Diyagrami Detay Rehberi

Bu dokuman, Eraser mimari diyagramindaki adimlarin kod karsiliklarini aciklar.
Amac: Diyagrami anlatirken her adimda hangi sinif ve hangi fonksiyonun calistigini hizli gormek.

## 1) Dataset Hazirlama

### 1.1 Goruntu Toplama
- Ne ise yarar:
Goruntuyu kameradan alip etiketleme kuyruguna ham veri olarak kaydeder.
- Nerede:
src/dataset/candidate_manager.py
- Sinif/Fonksiyon:
CandidateManager.process_frame
- Kisa aciklama:
Frame kalite analizinden gecer, uygun ise PNG ve JSON meta olarak kaydedilir, labeling_queue altina kopyalanir.

### 1.2 Annotation (Etiketleme)
- Ne ise yarar:
Operatorun bbox ve sinif bilgisini kaydetmesini saglar.
- Nerede:
src/api/routers/labeling.py, src/dataset/labeling_queue.py
- Sinif/Fonksiyon:
LabelingQueueManager.save_label, API endpoint: save_label
- Kisa aciklama:
Etiketler sample_id_labels.json dosyasina yazilir, aday status degeri labeled olur.

### 1.3 Dataset Split (Train/Validation/Test)
- Ne ise yarar:
Egitim, dogrulama ve test setlerini deterministik sekilde olusturur.
- Nerede:
src/training/dataset_splitter.py
- Sinif/Fonksiyon:
prepare_dataset, _split_counts, _prepare_output_dirs, _copy_pairs
- Kisa aciklama:
Veri dogrulanir, oranlara gore bolunur, kucuk veri setinde minimum valid/test sayisi korunur.

### 1.4 data.yaml Uretimi
- Ne ise yarar:
YOLO egitiminin hangi klasorleri okuyacagini ve sinif sayisini belirtir.
- Nerede:
src/training/dataset_splitter.py
- Sinif/Fonksiyon:
_write_data_yaml
- Kisa aciklama:
path, train, val, test, nc ve names alanlari yazilir.

## 2) Model Training

### 2.1 YOLO Training Process
- Ne ise yarar:
Modelin egitilmesini baslatir ve run klasorune artefakt uretir.
- Nerede:
src/training/train_yolo.py
- Sinif/Fonksiyon:
train
- Kisa aciklama:
Ultralytics YOLO nesnesi olusturulur, model.train ile egitim calisir.

### 2.2 Hyperparameters
- Ne ise yarar:
Egitim davranisini kontrol eder.
- Nerede:
config/training_config.yaml, src/training/train_yolo.py
- Sinif/Fonksiyon:
train icindeki config.get cagirilari
- Kisa aciklama:
epochs, imgsz, batch, optimizer, lr0, patience gibi degerler burada okunur.

### 2.3 Epoch Loop
- Ne ise yarar:
Modelin tum veriyi tekrar tekrar gorup agirliklari guncellemesini saglar.
- Nerede:
src/training/train_yolo.py (model.train), uretim logu: results.csv
- Sinif/Fonksiyon:
train fonksiyonu icinde model.train
- Kisa aciklama:
Her epoch, ileri yayilim + geri yayilim + optimizer adimi + validasyon adimlarini icerir.

### 2.4 Loss Calculation
- Ne ise yarar:
Modelin hatasini sayisallastirip ogrenmeyi yonlendirir.
- Nerede:
Ultralytics egitim motoru (dis kutuphane), cikti izleme: training_history.csv
- Sinif/Fonksiyon:
Dolayli olarak model.train ciktisi
- Kisa aciklama:
box_loss, cls_loss, dfl_loss degerleri dusmeye calisir.

### 2.5 Best.pt ve Last.pt
- Ne ise yarar:
Egitimdeki en iyi checkpoint ve son checkpoint dosyalarini saklar.
- Nerede:
data/training_runs/.../weights
- Sinif/Fonksiyon:
train, _select_best_checkpoint_by_map5095
- Kisa aciklama:
results.csv uzerinden en iyi val mAP50-95 epoch secilir, secili checkpoint kullanilir.

### 2.6 Performans Metrikleri
- Ne ise yarar:
Model kalitesini sayisal olarak olcer.
- Nerede:
src/training/train_yolo.py, src/training/test_evaluator.py
- Sinif/Fonksiyon:
_extract_best_row_metrics, evaluate_on_test_split
- Kisa aciklama:
Precision, Recall, mAP50, mAP50-95 ve F1 hesaplanir; test raporlari JSON/CSV yazilir.

## 3) Model Deployment

### 3.1 Trained Model (best.pt)
- Ne ise yarar:
Serviste kullanilacak modelin kaynagini belirler.
- Nerede:
scripts/train.py, src/training/export_onnx.py
- Sinif/Fonksiyon:
main, export
- Kisa aciklama:
Secilen pt modeli ONNX formatina donusturulur.

### 3.2 FastAPI Backend
- Ne ise yarar:
Kamera, etiketleme, dashboard, model ve inference API katmanini sunar.
- Nerede:
src/api/main.py
- Sinif/Fonksiyon:
FastAPI app, lifespan, _run_inference_loop
- Kisa aciklama:
Uygulama acilisinda model ve kamera hazirlanir, inference thread baslatilir.

### 3.3 REST API Endpointleri
- Ne ise yarar:
Istemci ile backend arasinda standart HTTP arayuzu saglar.
- Nerede:
src/api/routers/camera.py, src/api/routers/labeling.py, src/api/routers/inference.py, src/api/routers/dashboard.py
- Sinif/Fonksiyon:
list_cameras, capture_now, save_label, approve_candidate, inference_status, get_stats vb.
- Kisa aciklama:
Canli akis, manuel capture, etiketleme, durum ve metrik endpointleri bu katmandadir.

## 4) Client Workflow

### 4.1 Client Application
- Ne ise yarar:
Kullanicinin canli goruntu izlemesi, veri toplama ve sonuclari gormesi.
- Nerede:
src/dashboard/static/index.html, src/dashboard/static/capture.html, src/dashboard/static/app.js
- Sinif/Fonksiyon:
fetchStats, init, loadCameras, startCapture vb.
- Kisa aciklama:
Dashboard ve capture ekranlari API ile surekli veri alisverisi yapar.

### 4.2 Image Upload / Capture
- Ne ise yarar:
Kullanicinin o anki framei etikete aday olarak kaydetmesi.
- Nerede:
src/dashboard/static/capture.html, src/api/routers/camera.py
- Sinif/Fonksiyon:
runCaptureOnce, captureNow, API: capture_now
- Kisa aciklama:
Manuel tetikleme ile kamera framei labeling queueya eklenir.

### 4.3 HTTP Request
- Ne ise yarar:
Istemci-baglanti katmaninda veri tasir.
- Nerede:
src/dashboard/static/app.js, src/dashboard/static/capture.html
- Sinif/Fonksiyon:
fetchStats, fetchQueue, fetchCameraList, captureNow
- Kisa aciklama:
fetch ile JSON alinir/gonderilir.

### 4.4 API Authentication
- Ne ise yarar:
Isteklerin yetki kontrolu.
- Mevcut durum:
Bu projede zorunlu auth mekanizmasi su an uygulanmiyor.
- Not:
Kurumsal dagitimda JWT, API key veya reverse proxy auth eklenmeli.

## 5) Backend Processing

### 5.1 Request Validation
- Ne ise yarar:
Gelen verinin tip ve alan kurallarina uygunlugunu kontrol eder.
- Nerede:
src/api/routers/camera.py, src/api/routers/labeling.py, src/api/routers/inference.py
- Sinif/Fonksiyon:
Pydantic modelleri: CaptureNowRequest, LabelRequest, InferenceConfigRequest
- Kisa aciklama:
Yanlis tipte veri geldiginde istek erken asamada reddedilir.

### 5.2 Temporary Image Storage
- Ne ise yarar:
Islenecek goruntuyu dosya sisteminde saklar.
- Nerede:
src/dataset/candidate_manager.py
- Sinif/Fonksiyon:
CandidateManager.process_frame
- Kisa aciklama:
raw_captures ve labeling_queue klasorlerine dosya yazilir.

### 5.3 Image Loading
- Ne ise yarar:
Diskteki goruntuyu isleme alir.
- Nerede:
src/dataset/labeling_queue.py, src/inference/onnx_inference.py
- Sinif/Fonksiyon:
approve_candidate icinde cv2.imread, OnnxInference.predict
- Kisa aciklama:
Dosyadan veya streamden gelen frame bellekte numpy dizisine doner.

### 5.4 Image Preprocessing
- Ne ise yarar:
Modelin bekledigi tensor formati olusturur.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._preprocess
- Kisa aciklama:
Resize, renk donusumu ve normalize adimlari burada birlesir.

### 5.5 Image Resize
- Ne ise yarar:
Girdiyi modelin sabit boyutuna getirir.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._preprocess (cv2.resize)

### 5.6 Color Conversion
- Ne ise yarar:
OpenCV BGR formatini modelin bekledigi RGBye cevirir.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._preprocess (cv2.cvtColor)

### 5.7 Normalization
- Ne ise yarar:
Piksel degerlerini 0-1 araligina cekip numerik kararlilik saglar.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._preprocess

## 6) AI Inference Pipeline

### 6.1 YOLO Model Loading
- Ne ise yarar:
ONNX modelini bellekte InferenceSession olarak acmak.
- Nerede:
src/inference/model_loader.py, src/api/main.py
- Sinif/Fonksiyon:
ModelLoader.load_production_model, ModelLoader._load, ensure_inference_model_loaded

### 6.2 GPU / CPU / MPS Selection
- Ne ise yarar:
Hangi calisma saglayicisinin kullanilacagini secer.
- Nerede:
src/inference/model_loader.py
- Sinif/Fonksiyon:
_get_execution_providers
- Kisa aciklama:
CoreML varsa once onu, yoksa CPU provider secilir.

### 6.3 Inference
- Ne ise yarar:
Modelin goruntuden ham tahmin uretmesi.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference.predict

### 6.4 Confidence Threshold
- Ne ise yarar:
Dusuk guvenli tahminleri elemek.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference.__init__, OnnxInference._postprocess

### 6.5 IoU Threshold
- Ne ise yarar:
NMS asamasinda cakisan kutularin nasil elenecegini belirlemek.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference.__init__, OnnxInference._postprocess

### 6.6 NMS (Non-Maximum Suppression)
- Ne ise yarar:
Ayni nesne uzerindeki tekrarlayan kutulari azaltir.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._postprocess (cv2.dnn.NMSBoxes)

### 6.7 Object Detection ve Damage Classification
- Ne ise yarar:
Kutularin sinif ve guven skoru ile son tespit listesine donmesi.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
Detection veri sinifi, OnnxInference._postprocess

## 7) Post Processing

### 7.1 Bounding Box Generation
- Ne ise yarar:
Model cikti koordinatlarini orijinal goruntu boyutuna tasir.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._postprocess

### 7.2 Label Assignment
- Ne ise yarar:
Class id degerini sinif ismine cevirir.
- Nerede:
src/inference/onnx_inference.py
- Sinif/Fonksiyon:
OnnxInference._postprocess

### 7.3 Confidence Score
- Ne ise yarar:
Her tahminin guven seviyesini tutar ve raporlar.
- Nerede:
src/inference/onnx_inference.py, src/api/routers/dashboard.py
- Sinif/Fonksiyon:
Detection.confidence, record_detection

### 7.4 Visualization ve Annotated Image
- Ne ise yarar:
Kutularin canli goruntu uzerine cizilmesi.
- Nerede:
src/inference/result_processor.py
- Sinif/Fonksiyon:
ResultProcessor.process, ResultProcessor._draw_detections

## 8) Response

### 8.1 JSON Result
- Ne ise yarar:
Istemciye yorumlanabilir sonuc dondurmek.
- Nerede:
src/api/routers/dashboard.py, src/api/routers/camera.py, scripts/train.py
- Sinif/Fonksiyon:
get_stats, recent_detections, capture_now, main

### 8.2 Detection Coordinates / Confidence / Class Names
- Ne ise yarar:
Tespitin nerede ve ne kadar guvenli oldugunu belirtmek.
- Nerede:
src/inference/onnx_inference.py, src/api/routers/dashboard.py
- Sinif/Fonksiyon:
Detection, record_detection

### 8.3 Processed Image ve HTTP Response
- Ne ise yarar:
Annotate edilmis goruntuyu akista gostermek.
- Nerede:
src/api/routers/camera.py, src/pipeline/stream_manager.py
- Sinif/Fonksiyon:
video_stream, get_latest_annotated_frame

## 9) Monitoring ve Logging

### 9.1 API Logs
- Ne ise yarar:
Istek/yanit ve sistem olaylarini izlemek.
- Nerede:
src/utils/logger.py
- Sinif/Fonksiyon:
get_logger
- Kisa aciklama:
Hem konsol hem logs klasorune dosya logu yazilir.

### 9.2 Error Handling ve Exception Flow
- Ne ise yarar:
Hatalarin kontrollu ve acik mesajla donmesini saglar.
- Nerede:
src/api/routers/*.py, src/training/*.py
- Sinif/Fonksiyon:
HTTPException kullanan endpointler, RuntimeError raise edilen egitim fonksiyonlari

### 9.3 Processing Time
- Ne ise yarar:
Sure performansini olcmek.
- Nerede:
src/inference/onnx_inference.py, src/api/routers/dashboard.py
- Sinif/Fonksiyon:
InferenceResult.inference_time_ms, record_detection

### 9.4 Middleware Notu
- Ne ise yarar:
Request sure logu middleware sinifinda tanimli.
- Nerede:
src/api/middleware.py
- Sinif/Fonksiyon:
RequestLoggingMiddleware.dispatch
- Not:
main.py icinde middleware ekleme satiri su an gorunmedigi icin aktiflestirmek icin app.add_middleware ile kayitlanmasi gerekir.

## 10) Proje Yapisi (Kod Karsiliklari)

- Client katmani:
  - src/dashboard/static/index.html
  - src/dashboard/static/capture.html
  - src/dashboard/static/app.js
- FastAPI katmani:
  - src/api/main.py
  - src/api/routers/
- YOLO/Inference katmani:
  - src/inference/model_loader.py
  - src/inference/onnx_inference.py
  - src/inference/result_processor.py
- Egitim katmani:
  - src/training/train_yolo.py
  - src/training/dataset_splitter.py
  - src/training/test_evaluator.py
  - src/training/export_onnx.py
- Registry/Promotion:
  - src/registry/model_registry.py
  - src/registry/model_promoter.py
- Konfigurasyon:
  - config/app_config.yaml
  - config/training_config.yaml
  - config/inference_config.yaml
- Veri/Artefakt:
  - data/dataset/
  - data/training_runs/
  - data/models/registry/
  - data/models/production/
  - results/
  - logs/

## Yabanci Terimler Sozlugu (TR Karsilik + Neden Kullaniliyor)

- Dataset -> Veri kumesi
  - Neden: Egitim, dogrulama ve testte kullanilan etiketli orneklerin tamamini ifade eder.

- Annotation / Labeling -> Etiketleme
  - Neden: Goruntu icindeki nesnenin sinif ve konum bilgisini modele ogretmek icin gerekir.

- Split -> Bolme / Ayrirma
  - Neden: Modeli adil degerlendirmek icin veri train/valid/test olarak ayrilir.

- Hyperparameter -> Hiperparametre
  - Neden: Egitim davranisini onceden belirleyen, ogrenilmeyen ayarlardir.

- Epoch Loop -> Epok dongusu
  - Neden: Tum veri bir tur islendikce model agirliklari guncellenir; ogrenmenin temel dongusudur.

- Loss -> Kayip / Hata fonksiyonu
  - Neden: Tahmin ile gercek arasindaki farki sayisal olarak olcer, optimizer bu degeri azaltmaya calisir.

- Precision -> Kesinlik
  - Neden: Pozitif dediklerinin ne kadarinin dogru oldugunu gosterir.

- Recall -> Duyarlilik
  - Neden: Gercek pozitiflerin ne kadarini yakaladigini gosterir.

- mAP50 / mAP50-95 -> Ortalama dogruluk olcutu
  - Neden: Detection kalitesini IoU esiklerine gore toplu olcer, model karsilastirmada standarttir.

- Inference -> Cikarim / Tahmin calistirma
  - Neden: Egitilmis modelin yeni veride tahmin urettigi asamadir.

- Confidence Threshold -> Guven esigi
  - Neden: Cok dusuk guvenli tahminleri temizleyerek yanlis alarmi azaltir.

- IoU Threshold -> Kesisim oran esigi
  - Neden: Birbirine cok benzer kutulari NMS asamasinda elemek icin kullanilir.

- NMS (Non-Maximum Suppression) -> Maksimum olmayani bastirma
  - Neden: Ayni nesne uzerindeki tekrar kutulari azaltip temiz sonuc verir.

- Bounding Box -> Sinirlayici kutu
  - Neden: Nesnenin goruntu uzerindeki konumunu dikdortgenle ifade eder.

- Checkpoint -> Ara model kaydi
  - Neden: Egitimin belirli adimlarinda model agirliklarini saklayip geri donus saglar.

- Best.pt / Last.pt -> En iyi model / Son model
  - Neden: En iyi metrikli modeli ayri tutmak ve egitimin son durumunu korumak icin kullanilir.

- ONNX Export -> ONNX disa aktarma
  - Neden: Farkli runtime ortamlarda daha tasinabilir ve hizli calisan model formatina gecis icin.

- Registry -> Model kayit deposu
  - Neden: Model surumlerini metrik ve veri seti bilgisiyle takip etmeyi saglar.

- Promotion Gate -> Uretime gecis kapisi
  - Neden: Modelin belirli kalite esiklerini gecmeden productiona alinmasini engeller.

- Production -> Uretim ortami
  - Neden: Gercek kullanici trafigine hizmet veren canli ortamdir.

- Hot-swap -> Sicak model degisimi
  - Neden: Servisi durdurmadan model degistirebilmeyi saglar.

- API Endpoint -> API uc noktasi
  - Neden: Istemcinin backend fonksiyonlarina standart HTTP ile erismesi icin.

- Middleware -> Ara katman
  - Neden: Istek ve yanit akisina loglama, kimlik dogrulama gibi ortak davranis ekler.

## Kisa Ozet

Bu proje iki ana boru hattindan olusur:
- Egitim boru hatti: veri hazirlama -> egitim -> degerlendirme -> model kaydi -> promotion karari
- Inference boru hatti: istemci istegi/goruntu -> preprocess -> model cikarimi -> postprocess -> API ve dashboard yaniti

Diyagram sunumunda bu dosyayi referans alarak her kutuyu dogrudan kod seviyesine baglayabilirsiniz.
