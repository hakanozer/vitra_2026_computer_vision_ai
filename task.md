

Rol: Sen uzman bir python geliştiricisiniz ve görüntü işleme ve bilgisayarla görme alanında deneyimlisin var. Aşağıdaki görevleri yerine getirmeniz gerekiyor

Bağlam: 
- Python sürümüm: 3.11.9
- yolo: 8.2.63

Görev:
- Dashboard html içinde bir button olacak bu button'a tıklandığında yeni bir html sayfa açılacak bu sayfada kamera açılacak ve canlı olarak görüntü akışı sağlanacak.
- kullanıcının belirleyeceği saniye alabilen bir input olacak.
- Kamera listesi alınacak ve dropdown ile kullanıcıya sunulacak, kullanıcın seçtiği kamera ile belirlediğin saniye aralığında "http://0.0.0.0:8000/api/camera/capture-now" endpoint'ine istek atılacak ve canlı görüntü akışı sağlanacak.
- "camera_id": "camera-0", seçimine uygun olacak.
Kamera listesi endpoint:
curl -X 'GET' \
  'http://0.0.0.0:8000/api/camera/list' \
  -H 'accept: application/json'
Response body
{
  "camera-0": {
    "qsize": 0,
    "dropped": 0,
    "camera_alive": true
  }
}

Kemare görüntü yakalama:
curl -X 'POST' \
  'http://0.0.0.0:8000/api/camera/capture-now' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "camera_id": "camera-0",
  "source": "manual_capture",
  "force_add": true
}'
Response body
{
  "status": "captured",
  "sample_id": "32d14d1c-b105-4698-b489-3582f4eb1793",
  "camera_id": "camera-0",
  "image_url": "/api/labeling/image/32d14d1c-b105-4698-b489-3582f4eb1793"
}
- Görüntü listesi, kemare seçildikten ve örn her 5 sn bir görüntü alındıktan sonra gelen liste.

curl -X 'GET' \
  'http://0.0.0.0:8000/api/labeling/queue?limit=50' \
  -H 'accept: application/json'
{
  "candidates": [
    {
      "id": "e38d8241-c7d6-44b8-aec2-c6f895080983",
      "camera_id": "camera-0",
      "timestamp": 1784018195.463043,
      "source": "manual_capture",
      "quality": {
        "blur_score": 102.18140397947333,
        "brightness": 99.18911566840278,
        "contrast": 49.81238328853334,
        "is_acceptable": true
      },
      "image_url": "/labeling/image/e38d8241-c7d6-44b8-aec2-c6f895080983"
    }
  ]
}

Kısıt:
Yukardaki işlemler dışında herhangi bir şeye dokunma eğer yazılımda bu işlemler dışında farklı bir iş yapacaksan mutlaka bana sor.

Format:
- bir html dosya bu html dosyanın içerisindeki tema ile ders portun içindeki tema aynı olacak
- html'de üretmiş olduğun tasarım şu şekilde olacak sol tarafta kamera sağ tarafta saniye ve kamera seçimi bunun altında sol tarafta bir liste listeye tıklaıldığında daha önceden çekmiş olduğum listedeki seçilmiş olan görüntünün görünebileceği responsive bir tasarım
