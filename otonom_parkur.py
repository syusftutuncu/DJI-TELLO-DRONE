import cv2
import cv2.aruco as aruco
import numpy as np
from djitellopy import Tello
import time
import keyboard

# ==========================================
# GÖREV DURUMLARI (STATE MACHINE)
# ==========================================
DURUM_BEKLEME = 0
DURUM_KALKIS = 1
DURUM_KESIF = 2
DURUM_HEDEF_SEC = 3
DURUM_HIZALAN = 4
DURUM_BOSLUK_HIZALAN = 12
DURUM_KOR_UCUS = 6
DURUM_INIS = 7
DURUM_AKTIF_TARAMA = 8
DURUM_MAVI_H_ARA = 9
DURUM_MAVI_H_HIZALAN = 10
DURUM_FINAL_ARUCO_ARA = 11
mevcut_durum = DURUM_BEKLEME

# ==========================================
# PARKUR GÖREV SIRASI
# ==========================================
TOPLAM_ENGEL_SAYISI = 2
parkur_adimi = 0

# ==========================================
# HEDEF ARUCO ID'LERİ (SADECE BUNLARA ODAKLANIR)
# ==========================================
HEDEF_ARUCO_IDLER = [157, 158]

# ==========================================
# AGRESİF ARUCO DETEKTÖR AYARLARI
# ==========================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)


def aruco_parametrelerini_hazirla():
    try:
        params = aruco.DetectorParameters_create()
    except AttributeError:
        params = aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 5
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.errorCorrectionRate = 0.6
    return params


aruco_params = {
    "params": aruco_parametrelerini_hazirla(),
    "detector": None
}

if hasattr(aruco, "ArucoDetector"):
    aruco_params["detector"] = aruco.ArucoDetector(aruco_dict, aruco_params["params"])


def aruco_oku(frame):
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if aruco_params["detector"] is not None:
        koseler, idler, _ = aruco_params["detector"].detectMarkers(gri)
    else:
        koseler, idler, _ = aruco.detectMarkers(gri, aruco_dict, parameters=aruco_params["params"])

    gecerli_koseler = []
    gecerli_idler = []

    if idler is not None:
        for i, a_id in enumerate(idler.flatten()):
            if a_id in HEDEF_ARUCO_IDLER:
                gecerli_koseler.append(koseler[i])
                gecerli_idler.append([a_id])

        if len(gecerli_idler) > 0:
            return tuple(gecerli_koseler), np.array(gecerli_idler, dtype=np.int32)

    return None, None


# ==========================================
# İÇ ODAKLI MESAFE DÖNÜŞÜMÜ (EN BÜYÜK BOŞLUK ODAKLI)
# ==========================================
bosluk_takip = {"cx": None, "cy": None, "aruco_tepe_y": None}

# Boşluk arama bölgesi (ROI). Ana döngüdeki görselleştirme kutusu da bunu kullanır.
BOSLUK_ROI = {"x_bas": 100, "x_bit": 860, "y_bas": 90, "y_bit": 630}


def bosluk_hafizasini_sifirla():
    bosluk_takip["cx"] = None
    bosluk_takip["cy"] = None
    bosluk_takip["aruco_tepe_y"] = None


def guvenli_bosluk_bul(frame):
    # =========================================================
    # DÜZELTME 1 (EN BÜYÜK BOŞLUK): Arama bölgesi (ROI) genişletildi.
    # Eski dar ROI (200-760 / 140-580) yalnızca karenin merkezini
    # tarıyordu. Engelin KENARINDAKİ büyük boşluk ROI sınırında
    # kırpıldığı için küçük ölçülüyor, MERKEZE yakın küçük boşluk tam
    # ölçülüp "kazanıyordu". Bu yüzden drone en büyük değil, kendine en
    # yakın (çoğunlukla en küçük) boşluğu seçiyordu.
    # =========================================================
    x_bas, x_bit = BOSLUK_ROI["x_bas"], BOSLUK_ROI["x_bit"]
    y_bas, y_bit = BOSLUK_ROI["y_bas"], BOSLUK_ROI["y_bit"]

    aruco_koseler, _ = aruco_oku(frame)
    if aruco_koseler is not None and len(aruco_koseler) > 0:
        min_y = frame.shape[0]
        for kose in aruco_koseler:
            for nokta in kose[0]:
                if nokta[1] < min_y:
                    min_y = nokta[1]
        bosluk_takip["aruco_tepe_y"] = int(min_y)

    roi = frame[y_bas:y_bit, x_bas:x_bit]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    alt_siyah = np.array([0, 0, 0])
    ust_siyah = np.array([180, 255, 55])
    maske_siyah = cv2.inRange(hsv, alt_siyah, ust_siyah)

    alt_kirmizi1 = np.array([0, 35, 40])
    ust_kirmizi1 = np.array([15, 255, 255])
    alt_kirmizi2 = np.array([165, 35, 40])
    ust_kirmizi2 = np.array([180, 255, 255])
    maske_kirmizi1 = cv2.inRange(hsv, alt_kirmizi1, ust_kirmizi1)
    maske_kirmizi2 = cv2.inRange(hsv, alt_kirmizi2, ust_kirmizi2)
    maske_kirmizi = cv2.bitwise_or(maske_kirmizi1, maske_kirmizi2)

    maske_engel_ham = cv2.bitwise_or(maske_siyah, maske_kirmizi)

    kernel = np.ones((7, 7), np.uint8)
    maske_engel_ham = cv2.morphologyEx(maske_engel_ham, cv2.MORPH_OPEN, kernel, iterations=2)
    maske_engel_ham = cv2.morphologyEx(maske_engel_ham, cv2.MORPH_CLOSE, kernel, iterations=2)

    MIN_KONTUR_ALAN = 250
    konturlar, _ = cv2.findContours(maske_engel_ham, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maske_engel = np.zeros_like(maske_engel_ham)
    toplam_alan = 0
    gecerli_konturlar = []

    for k in konturlar:
        alan = cv2.contourArea(k)
        if alan >= MIN_KONTUR_ALAN:
            gecerli_konturlar.append(k)
            cv2.drawContours(maske_engel, [k], -1, 255, -1)
            toplam_alan += alan

    cx = 960 // 2
    cy = 720 // 2

    BANT_ALAN_ESIGI = 900
    if toplam_alan > BANT_ALAN_ESIGI and len(gecerli_konturlar) > 0:
        # =========================================================
        # DÜZELTME 5 (ÇERÇEVE İÇİ + YAKINDA DA ÇALIŞSIN):
        # Önceki "tek en büyük kontur" yöntemi, drone engele yaklaşıp
        # çerçeve ROI kenarlarından TAŞINCA çerçeveyi tek parça göremiyordu
        # (üst/alt/yan çubuklar ayrı konturlara bölünüyordu). Tek çubuğu
        # doldurmak iç boşluk bırakmadığı için sürekli bant_var=False
        # dönüyor, drone boşluğa hiç bakmadan direkt kör uçuşa geçiyordu.
        #
        # Çözüm: Çerçeveyi KIRMIZI maskeden tanımla. Kırmızının büyük
        # parçalarının dışbükey zarfı (convex hull), çubuklar bölünse/
        # kırpılsa bile çerçevenin çevrelediği alanı yeniden birleştirir.
        # Kırmızı kullanıldığı için arka plandaki koyu lekeler bölgeyi
        # şişirmez -> çerçeve DIŞI yerler yine boşluk olarak seçilmez.
        # =========================================================
        maske_kirmizi_temiz = cv2.morphologyEx(maske_kirmizi, cv2.MORPH_OPEN, kernel, iterations=1)
        maske_kirmizi_temiz = cv2.morphologyEx(maske_kirmizi_temiz, cv2.MORPH_CLOSE, kernel, iterations=2)
        kirmizi_konturlar, _ = cv2.findContours(maske_kirmizi_temiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kirmizi_buyuk = [k for k in kirmizi_konturlar if cv2.contourArea(k) >= MIN_KONTUR_ALAN]

        maske_cerceve_ici = np.zeros_like(maske_engel)

        if len(kirmizi_buyuk) > 0:
            # Kırmızı çerçevenin (parçalı/kırpık olsa da) çevrelediği alan.
            kirmizi_noktalar = np.vstack(kirmizi_buyuk)
            hull = cv2.convexHull(kirmizi_noktalar)
            cv2.drawContours(maske_cerceve_ici, [hull], -1, 255, -1)
        else:
            # Kırmızı görünmüyorsa: en büyük engel konturunu (çerçeveyi) doldur.
            cerceve_konturu = max(gecerli_konturlar, key=cv2.contourArea)
            cv2.drawContours(maske_cerceve_ici, [cerceve_konturu], -1, 255, -1)

        # Çerçeve bölgesi anlamlı bir alan kaplamıyorsa geçiş arama.
        if cv2.countNonZero(maske_cerceve_ici) < 3000:
            bosluk_hafizasini_sifirla()
            return cx, cy, False, None

        guvenlik_kernel = np.ones((25, 25), np.uint8)
        maske_engel_sisirilmis = cv2.dilate(maske_engel, guvenlik_kernel, iterations=1)

        # Çerçeve içi - (şişirilmiş engel) = çerçeve içindeki güvenli boşluk.
        bosluk_maskesi = cv2.bitwise_and(maske_cerceve_ici, cv2.bitwise_not(maske_engel_sisirilmis))

        # =========================================================
        # DÜZELTME 2 (GÜVENLİ BÖLGEYE BAKMADAN GEÇME):
        # ArUco kırpması artık maskenin TAMAMINI silmiyor.
        # Eski kodda ArUco kareye yaklaşınca (tepe_y ROI üstünün üzerine
        # çıkınca) 'bosluk_maskesi[:] = 0' çalışıyor, r=0 oluyor ve
        # bant_var=False dönüyordu. Sonuç: drone boşluğa hiç hizalanmadan
        # 15 kare sonra direkt kör uçuşa geçiyordu.
        # Artık kırpma yalnızca ardından YETERİNCE boşluk kalıyorsa
        # uygulanır; kalmıyorsa kırpma atlanır (maske silinmez).
        # =========================================================
        aruco_y = bosluk_takip.get("aruco_tepe_y")
        if aruco_y is not None:
            guvenli_sinir_y = aruco_y - y_bas - 10
            if 0 < guvenli_sinir_y < (y_bit - y_bas):
                aday_maske = bosluk_maskesi.copy()
                aday_maske[guvenli_sinir_y:, :] = 0
                if cv2.countNonZero(aday_maske) > 400:
                    bosluk_maskesi = aday_maske
            # guvenli_sinir_y <= 0 durumunda maskeyi komple SİLMİYORUZ (eski hata).

        mesafe_haritasi = cv2.distanceTransform(bosluk_maskesi, cv2.DIST_L2, 5)

        # =========================================================
        # DÜZELTME 3 (EN BÜYÜK BOŞLUĞU AÇIKÇA SEÇ):
        # Boşluk maskesini bağlı bileşenlere ayırıp her ayrı boşluğun
        # içine sığan en büyük dairenin yarıçapını (distanceTransform tepe
        # değeri) buluyoruz. Yarıçapı EN BÜYÜK olan boşluğu seçiyoruz.
        # Böylece "merkeze en yakın" değil, gerçekten "en geniş geçiş"
        # noktası hedeflenir.
        # =========================================================
        sayi, etiketler, istatistik, _ = cv2.connectedComponentsWithStats(bosluk_maskesi, connectivity=8)

        en_iyi_yaricap = 0.0
        en_iyi_x_roi = None
        en_iyi_y_roi = None

        for etiket in range(1, sayi):
            if istatistik[etiket, cv2.CC_STAT_AREA] < 150:
                continue
            bilesen_dt = np.where(etiketler == etiket, mesafe_haritasi, 0)
            _, yerel_max, _, yerel_loc = cv2.minMaxLoc(bilesen_dt)
            if yerel_max > en_iyi_yaricap:
                en_iyi_yaricap = yerel_max
                en_iyi_x_roi, en_iyi_y_roi = yerel_loc

        MIN_GECIS_YARICAPI = 15
        if en_iyi_x_roi is None or en_iyi_yaricap < MIN_GECIS_YARICAPI:
            bosluk_hafizasini_sifirla()
            return frame.shape[1] // 2, frame.shape[0] // 2, False, None

        r = int(en_iyi_yaricap)

        cx = en_iyi_x_roi + x_bas
        cy = en_iyi_y_roi + y_bas

        alpha = 0.35
        if bosluk_takip["cx"] is None:
            bosluk_takip["cx"], bosluk_takip["cy"] = cx, cy
        else:
            # Seçilen boşluk bir öncekinden çok uzaktaysa (yani daha büyük
            # bir boşluğa geçilmişse) yumuşatma hafızasını sıfırla; iki
            # boşluğun ortasına "karışık" bir hedef üretmesin.
            if abs(cx - bosluk_takip["cx"]) > 130 or abs(cy - bosluk_takip["cy"]) > 130:
                bosluk_takip["cx"], bosluk_takip["cy"] = cx, cy
            else:
                cx = int(alpha * cx + (1 - alpha) * bosluk_takip["cx"])
                cy = int(alpha * cy + (1 - alpha) * bosluk_takip["cy"])
                bosluk_takip["cx"], bosluk_takip["cy"] = cx, cy

        kutu_verisi = (cx - r, cy - r, r * 2, r * 2)
        return cx, cy, True, kutu_verisi

    bosluk_hafizasini_sifirla()
    return cx, cy, False, None


# ==========================================
# H HARFİ DETEKSİYONU (GERÇEK MAVİ RENGİ)
# ==========================================
def mavi_h_tespit_et(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    alt_renk = np.array([90, 80, 50])
    ust_renk = np.array([130, 255, 255])

    maske = cv2.inRange(hsv, alt_renk, ust_renk)

    kernel_open = np.ones((5, 5), np.uint8)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel_open)
    maske = cv2.dilate(maske, np.ones((5, 5), np.uint8), iterations=2)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    en_iyi_h_konturu = None
    en_iyi_alan = 0

    if konturlar:
        konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)

        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            if alan < 1500:
                continue

            x, y, w, h = cv2.boundingRect(kontur)
            en_boy_orani = float(w) / h
            hull = cv2.convexHull(kontur)
            hull_alani = cv2.contourArea(hull)
            doluluk_orani = float(alan) / hull_alani if hull_alani > 0 else 0

            if 0.3 < en_boy_orani < 2.2 and 0.20 < doluluk_orani < 0.65:
                en_iyi_h_konturu = kontur
                en_iyi_alan = alan
                break

    if en_iyi_h_konturu is not None:
        M = cv2.moments(en_iyi_h_konturu)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, en_iyi_alan, en_iyi_h_konturu

    return None, None, 0, None


# ==========================================
# ANA OTONOM UÇUŞ DÖNGÜSÜ
# ==========================================
def main():
    global mevcut_durum, parkur_adimi
    drone = Tello()

    try:
        print("[SİSTEM] Drone'a bağlanılıyor...")
        drone.connect()
        pil_seviyesi = drone.get_battery()
        print(f"[BİLGİ] Batarya Seviyesi: %{pil_seviyesi}")

        if pil_seviyesi < 15:
            print("[UYARI] Batarya düşük! İptal edildi.")
            return

        drone.streamon()
        print("[SİSTEM] Kamera açıldı. Kalkış için 'T' tuşuna basın.")

        frame_merkez_x = 960 // 2
        frame_merkez_y = 720 // 2

        onay_sayaci = 0
        HEDEF_ONAY_SINIRI = 5
        hizalanma_onay_sayaci = 0
        bosluk_onay_sayaci = 0
        bant_yok_sayaci = 0
        hedef_bulunamadi_sayaci = 0
        tarama_baslangic_zamani = 0
        tarama_fazi = 0
        final_arama_fazi = 0
        final_arama_faz_baslangic = 0
        aranan_id = None
        h_hizalanma_asama = 0
        h_ileri_baslangic = 0
        h_tarama_baslangic = 0
        son_hedef_yonu = 0
        kesif_irtifasi = 60
        kesif_baslangic_zamani = 0

        while True:
            if keyboard.is_pressed('e'):
                print("\n[ACİL DURUM] İniş yapılıyor...")
                drone.land()
                break

            img = drone.get_frame_read().frame
            if img is None:
                continue

            img = cv2.resize(img, (960, 720))

            cv2.line(img, (frame_merkez_x - 10, frame_merkez_y), (frame_merkez_x + 10, frame_merkez_y), (255, 255, 255),
                     1)
            cv2.line(img, (frame_merkez_x, frame_merkez_y - 10), (frame_merkez_x, frame_merkez_y + 10), (255, 255, 255),
                     1)

            if parkur_adimi < TOPLAM_ENGEL_SAYISI and mevcut_durum < DURUM_MAVI_H_ARA:
                hedef_metni = f"{aranan_id}" if aranan_id is not None else "EN YAKIN (157 VEYA 158)"
                cv2.putText(img, f"GOREV: HEDEF -> {hedef_metni}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
            elif mevcut_durum == DURUM_FINAL_ARUCO_ARA:
                yon_str = "SOL" if son_hedef_yonu == -1 else ("SAG" if son_hedef_yonu == 1 else "CIFT YON")
                cv2.putText(img, f"GOREV: ENGEL SONRASI ARUCO ARAMASI ({yon_str})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 165, 255), 2)
            elif mevcut_durum >= DURUM_MAVI_H_ARA:
                cv2.putText(img, "GOREV: FINAL H INISI AKTIF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255),
                            2)
            else:
                cv2.putText(img, "PARKUR TAMAMLANDI!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if mevcut_durum == DURUM_BEKLEME:
                cv2.putText(img, "KAMERA AKTIF - KALKIS ICIN 'T' TUSUNA BASIN", (120, 360), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2)
                if keyboard.is_pressed('t'):
                    mevcut_durum = DURUM_KALKIS

            elif mevcut_durum == DURUM_KALKIS:
                cv2.putText(img, f"HAVALANILIYOR... PIL: %{drone.get_battery()}", (250, 360), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                cv2.imshow("Otonom Parkur", img)
                cv2.waitKey(1)
                drone.takeoff()
                print("[SİSTEM] Kalkış yapıldı. Stabilizasyon bekleniyor...")
                time.sleep(2.0)
                kesif_baslangic_zamani = time.time()
                mevcut_durum = DURUM_KESIF

            elif mevcut_durum == DURUM_KESIF:
                mevcut_yukseklik = drone.get_height()
                if mevcut_yukseklik <= 10: mevcut_yukseklik = 110
                hata_y = kesif_irtifasi - mevcut_yukseklik
                cv2.putText(img, f"{kesif_irtifasi}CM IRTIFAYA GIDILIYOR | MEVCUT: {mevcut_yukseklik}cm", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if time.time() - kesif_baslangic_zamani > 4.0:
                    hata_y = 0

                if abs(hata_y) > 10:
                    drone.send_rc_control(0, 0, max(-30, min(30, int(hata_y * 1.5))), 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.5)
                    aranan_id = None
                    onay_sayaci = 0
                    hizalanma_onay_sayaci = 0
                    hedef_bulunamadi_sayaci = 0
                    tarama_fazi = 0
                    mevcut_durum = DURUM_HEDEF_SEC

            elif mevcut_durum == DURUM_HEDEF_SEC:
                drone.send_rc_control(0, 0, 0, 0)
                aruco_koseler, aruco_idler = aruco_oku(img)
                dogru_hedef_bulundu = False
                bulunan_kose = None

                if aruco_idler is not None and len(aruco_idler) > 0:
                    aruco.drawDetectedMarkers(img, aruco_koseler, aruco_idler)

                    en_buyuk_alan = 0
                    en_yakin_index = -1

                    for i, a_id in enumerate(aruco_idler.flatten()):
                        if aranan_id is None:
                            alan = cv2.contourArea(aruco_koseler[i])
                            if alan > en_buyuk_alan:
                                en_buyuk_alan = alan
                                en_yakin_index = i
                        else:
                            if a_id == aranan_id:
                                en_yakin_index = i
                                break

                    if en_yakin_index != -1:
                        if aranan_id is None:
                            aranan_id = int(aruco_idler[en_yakin_index][0])
                        dogru_hedef_bulundu = True
                        bulunan_kose = aruco_koseler[en_yakin_index]

                if dogru_hedef_bulundu:
                    if onay_sayaci == 0:
                        get_merkez_x = int(np.mean(bulunan_kose[0][:, 0]))
                        hata_x = get_merkez_x - frame_merkez_x
                        if hata_x < -160:
                            son_hedef_yonu = -1
                        elif hata_x > 160:
                            son_hedef_yonu = 1
                        else:
                            son_hedef_yonu = 0
                    onay_sayaci += 1
                    hedef_bulunamadi_sayaci = 0
                else:
                    hedef_bulunamadi_sayaci += 1
                    cv2.putText(img, f"GECERLI ARUCO (157/158) ARANIYOR... ({hedef_bulunamadi_sayaci}/45)", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if onay_sayaci > 0:
                    cv2.putText(img, f"HEDEF {aranan_id} ONAYI: {onay_sayaci}/{HEDEF_ONAY_SINIRI}", (100, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 3)

                if onay_sayaci >= HEDEF_ONAY_SINIRI:
                    hedef_bulunamadi_sayaci = 0
                    mevcut_durum = DURUM_HIZALAN

                if hedef_bulunamadi_sayaci > 45:
                    mevcut_durum = DURUM_AKTIF_TARAMA
                    tarama_fazi = 0
                    hedef_bulunamadi_sayaci = 0
                    onay_sayaci = 0
                    tarama_baslangic_zamani = time.time()

            elif mevcut_durum == DURUM_HIZALAN:
                aruco_koseler, aruco_idler = aruco_oku(img)
                hedef_bulundu = False
                hata_x = 0
                hata_y = 0
                alan = 0
                hedef_index = -1

                if aruco_idler is not None and len(aruco_idler) > 0:
                    for i, a_id in enumerate(aruco_idler.flatten()):
                        if a_id == aranan_id:
                            aruco.drawDetectedMarkers(img, aruco_koseler, aruco_idler)
                            alan = cv2.contourArea(aruco_koseler[i])

                            aruco_merkez_x = int(np.mean(aruco_koseler[i][0][:, 0]))
                            aruco_merkez_y = int(np.mean(aruco_koseler[i][0][:, 1]))

                            hata_x = aruco_merkez_x - frame_merkez_x
                            hata_y = frame_merkez_y - aruco_merkez_y

                            hedef_bulundu = True
                            hedef_index = i
                            break

                if hedef_bulundu:
                    hedef_bulunamadi_sayaci = 0
                    koseler = aruco_koseler[hedef_index][0]
                    sol_ust, sag_ust, sag_alt, sol_alt = koseler[0], koseler[1], koseler[2], koseler[3]
                    sol_kenar = np.linalg.norm(sol_ust - sol_alt)
                    sag_kenar = np.linalg.norm(sag_ust - sag_alt)
                    perspektif_orani = (sol_kenar - sag_kenar) / max(sol_kenar, sag_kenar)

                    yaw_hiz = max(-40, min(40, int(hata_x / 3)))
                    sag_sol_hiz = max(-40, min(40, int(perspektif_orani * 250)))

                    if abs(hata_y) < 20:
                        yukari_asagi_hiz = 0
                    else:
                        yukari_asagi_hiz = max(-15, min(15, int(hata_y / 6)))

                    hedef_alan = 6000
                    alan_hatasi = hedef_alan - alan
                    aktif_ileri_hiz = int(alan_hatasi * 0.015)

                    if abs(perspektif_orani) > 0.08:
                        ileri_hiz = max(-15, min(15, aktif_ileri_hiz))
                        cv2.putText(img, "YORUNGE (ORBIT) CIZILIYOR...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 165, 255), 2)
                        hizalanma_onay_sayaci = 0
                    else:
                        ileri_hiz = max(-20, min(20, aktif_ileri_hiz))

                        if alan < 5000:
                            cv2.putText(img, "TAM KARSISINDA - YAKLASILIYOR...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 0), 2)
                        elif alan > 7000:
                            cv2.putText(img, "COK YAKIN - GERI FRENLEME...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)
                        else:
                            ileri_hiz = 0
                            cv2.putText(img, "MESAFE IYI - BEKLENIYOR...", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 0), 2)

                        if abs(hata_x) < 20 and abs(hata_y) < 25 and abs(
                                perspektif_orani) < 0.04 and 5000 <= alan <= 7000:
                            hizalanma_onay_sayaci += 1
                            cv2.putText(img, f"TAM MERKEZ KILITLENME! ({hizalanma_onay_sayaci}/15)", (30, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            hizalanma_onay_sayaci = 0

                    if hizalanma_onay_sayaci >= 15:
                        drone.send_rc_control(0, 0, 0, 0)
                        print("[SİSTEM] ArUco merkezine kilitlenildi. Boşluk analizine geçiliyor.")
                        time.sleep(0.5)
                        bosluk_hafizasini_sifirla()
                        mevcut_durum = DURUM_BOSLUK_HIZALAN
                        bosluk_onay_sayaci = 0
                        bant_yok_sayaci = 0
                    else:
                        drone.send_rc_control(sag_sol_hiz, ileri_hiz, yukari_asagi_hiz, yaw_hiz)
                else:
                    hedef_bulunamadi_sayaci += 1
                    cv2.putText(img, f"HIZALANMA ICIN ARUCO ARANIYOR... ({hedef_bulunamadi_sayaci}/45)", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    drone.send_rc_control(0, 0, 0, 0)

                    if hedef_bulunamadi_sayaci > 15:
                        mevcut_durum = DURUM_HEDEF_SEC
                        hedef_bulunamadi_sayaci = 0
                        onay_sayaci = 0

            elif mevcut_durum == DURUM_BOSLUK_HIZALAN:
                bosluk_x, bosluk_y, bant_var, kutu = guvenli_bosluk_bul(img)

                cv2.rectangle(img, (BOSLUK_ROI["x_bas"], BOSLUK_ROI["y_bas"]),
                              (BOSLUK_ROI["x_bit"], BOSLUK_ROI["y_bit"]), (255, 0, 0), 2)

                if not bant_var:
                    bant_yok_sayaci += 1
                    cv2.putText(img, f"BANT ARANIYOR, BEKLEYIN ({bant_yok_sayaci}/25)", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    drone.send_rc_control(0, 0, 0, 0)

                    # DÜZELTME: kör uçuşa geçmeden önce boşluğa hizalanmaya daha
                    # fazla şans ver (15 -> 25). Boşluk gerçekten yoksa ancak o
                    # zaman doğrudan geçilir.
                    if bant_yok_sayaci >= 25:
                        cv2.putText(img, "SIYAH BANT YOK: DIREKT GECILIYOR", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                        mevcut_durum = DURUM_KOR_UCUS
                        bant_yok_sayaci = 0

                    continue

                bant_yok_sayaci = 0

                if bant_var and kutu is not None:
                    bx, by, bw, bh = kutu
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(img, "BANTLI ENGEL: EN GENIS PASAJA YONELINIYOR", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 165, 255), 2)

                cv2.circle(img, (bosluk_x, bosluk_y), 6, (0, 255, 0), -1)

                hedef_y = frame_merkez_y - 10
                cv2.circle(img, (frame_merkez_x, hedef_y), 7, (255, 0, 255), -1)

                hata_x = bosluk_x - frame_merkez_x
                hata_y = bosluk_y - hedef_y

                # =========================================================
                # TOLERANS: 40 piksel (yaklaşık 1-2 cm pay).
                # =========================================================
                if abs(hata_x) < 40:
                    sag_sol_hiz = 0
                else:
                    sag_sol_hiz = int(hata_x * 0.10)
                    if 0 < sag_sol_hiz < 10:
                        sag_sol_hiz = 10
                    elif -10 < sag_sol_hiz < 0:
                        sag_sol_hiz = -10

                if abs(hata_y) < 40:
                    yukari_asagi_hiz = 0
                else:
                    yukari_asagi_hiz = int(-hata_y * 0.12)
                    if 0 < yukari_asagi_hiz < 10:
                        yukari_asagi_hiz = 10
                    elif -10 < yukari_asagi_hiz < 0:
                        yukari_asagi_hiz = -10

                sag_sol_hiz = max(-15, min(15, sag_sol_hiz))
                yukari_asagi_hiz = max(-15, min(15, yukari_asagi_hiz))

                # Onay mekanizması da bu 40 piksellik esneklikle çalışır.
                if abs(hata_x) < 40 and abs(hata_y) < 40 and sag_sol_hiz == 0 and yukari_asagi_hiz == 0:
                    bosluk_onay_sayaci += 1
                    cv2.putText(img, f"STABIL KILITLENME SAGLANDI! ({bosluk_onay_sayaci}/10)", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    drone.send_rc_control(0, 0, 0, 0)

                    if bosluk_onay_sayaci >= 10:
                        mevcut_durum = DURUM_KOR_UCUS
                        bosluk_onay_sayaci = 0
                else:
                    bosluk_onay_sayaci = 0
                    drone.send_rc_control(sag_sol_hiz, 0, yukari_asagi_hiz, 0)

            elif mevcut_durum == DURUM_KOR_UCUS:
                cv2.putText(img, "HIZALANDI - MERKEZDE BEKLENIYOR...", (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 3)
                cv2.imshow("Otonom Parkur", img)
                cv2.waitKey(1)

                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(1.5)

                cv2.putText(img, "ENGEL ICINDEN GECILIYOR...", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),
                            3)
                cv2.imshow("Otonom Parkur", img)
                cv2.waitKey(1)

                kor_ucus_suresi = 2.9 if parkur_adimi == 0 else 2.8

                drone.send_rc_control(0, 35, -10, 0)
                time.sleep(kor_ucus_suresi)

                drone.send_rc_control(0, 0, 0, 0)

                parkur_adimi += 1

                if parkur_adimi >= TOPLAM_ENGEL_SAYISI:
                    mevcut_durum = DURUM_FINAL_ARUCO_ARA
                    final_arama_fazi = 0
                else:
                    kesif_irtifasi = 105
                    onay_sayaci = 0
                    hedef_bulunamadi_sayaci = 0
                    kesif_baslangic_zamani = time.time()
                    mevcut_durum = DURUM_KESIF

            elif mevcut_durum == DURUM_AKTIF_TARAMA:
                mevcut_yukseklik = drone.get_height()
                aruco_koseler, aruco_idler = aruco_oku(img)
                dogru_hedef_bulundu = False

                if aruco_idler is not None and len(aruco_idler) > 0:
                    for i, a_id in enumerate(aruco_idler.flatten()):
                        if a_id == aranan_id: dogru_hedef_bulundu = True; break

                if dogru_hedef_bulundu:
                    drone.send_rc_control(0, 0, 0, 0)
                    mevcut_durum = DURUM_HEDEF_SEC
                    hedef_bulunamadi_sayaci = 0
                    tarama_fazi = 0
                else:
                    if tarama_fazi == 0:
                        cv2.putText(img, "TARAMA 1/3: YUKSEKTE CEVRE KONTROLU (SAGA/SOLA)", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        gecen_sure = time.time() - tarama_baslangic_zamani
                        if gecen_sure < 3.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        elif gecen_sure < 9.0:
                            drone.send_rc_control(0, 0, 0, 35)
                        elif gecen_sure < 12.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            tarama_fazi = 1
                            tarama_baslangic_zamani = time.time()

                    elif tarama_fazi == 1:
                        hedef_y = 45
                        hata_y_tarama = hedef_y - mevcut_yukseklik
                        cv2.putText(img, f"TARAMA 2/3: {hedef_y}CM'E INILIYOR", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 165, 255), 2)

                        if time.time() - tarama_baslangic_zamani > 3.5:
                            hata_y_tarama = 0

                        if abs(hata_y_tarama) > 10:
                            drone.send_rc_control(0, 0, max(-30, min(30, int(hata_y_tarama * 1.5))), 0)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            tarama_fazi = 2
                            tarama_baslangic_zamani = time.time()

                    elif tarama_fazi == 2:
                        cv2.putText(img, "TARAMA 3/3: ALCAKTA (55cm) CEVRE KONTROLU", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        gecen_sure = time.time() - tarama_baslangic_zamani
                        if gecen_sure < 3.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        elif gecen_sure < 9.0:
                            drone.send_rc_control(0, 0, 0, 35)
                        elif gecen_sure < 12.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            print("[SİSTEM] İki kademeli ArUco taraması başarısız. Final H aramasına geçiliyor.")
                            mevcut_durum = DURUM_MAVI_H_ARA
                            tarama_fazi = 0
                            h_tarama_baslangic = 0

            elif mevcut_durum == DURUM_FINAL_ARUCO_ARA:
                mevcut_yukseklik = drone.get_height()
                aruco_koseler, aruco_idler = aruco_oku(img)

                if aruco_idler is not None and len(aruco_idler) > 0:
                    drone.send_rc_control(0, 0, 0, 0)
                    aranan_id = int(aruco_idler[0][0])
                    mevcut_durum = DURUM_HEDEF_SEC
                    hedef_bulunamadi_sayaci = 0
                    onay_sayaci = 0
                else:
                    if final_arama_fazi == 0:
                        hata_y_final = 110 - mevcut_yukseklik
                        cv2.putText(img, f"FINAL TARAMA 1/4: 110CM'E CIKILIYOR", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        if abs(hata_y_final) > 10:
                            drone.send_rc_control(0, 0, max(-35, min(35, int(hata_y_final * 1.5))), 0)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            final_arama_fazi = 1
                            final_arama_faz_baslangic = time.time()

                    elif final_arama_fazi == 1:
                        kalan_sure = time.time() - final_arama_faz_baslangic
                        cv2.putText(img, "FINAL TARAMA 2/4: 110CM CEVRE KONTROLU", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        if kalan_sure < 3.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        elif kalan_sure < 9.0:
                            drone.send_rc_control(0, 0, 0, 35)
                        elif kalan_sure < 12.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            final_arama_fazi = 2

                    elif final_arama_fazi == 2:
                        hata_y_final = 55 - mevcut_yukseklik
                        cv2.putText(img, f"FINAL TARAMA 3/4: 55CM'E INILIYOR", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 165, 255), 2)
                        if abs(hata_y_final) > 10:
                            drone.send_rc_control(0, 0, max(-35, min(35, int(hata_y_final * 1.5))), 0)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            final_arama_fazi = 3
                            final_arama_faz_baslangic = time.time()

                    elif final_arama_fazi == 3:
                        kalan_sure = time.time() - final_arama_faz_baslangic
                        cv2.putText(img, "FINAL TARAMA 4/4: 55CM CEVRE KONTROLU", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        if kalan_sure < 3.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        elif kalan_sure < 9.0:
                            drone.send_rc_control(0, 0, 0, 35)
                        elif kalan_sure < 12.0:
                            drone.send_rc_control(0, 0, 0, -35)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            mevcut_durum = DURUM_MAVI_H_ARA
                            h_tarama_baslangic = 0

            elif mevcut_durum == DURUM_MAVI_H_ARA:
                mevcut_yukseklik = drone.get_height()

                hata_y_h = 25 - mevcut_yukseklik
                dikey_hiz = max(-35, min(35, int(hata_y_h * 1.5)))

                cv2.putText(img, f"25CM'E ALCALINIYOR | MEVCUT: {mevcut_yukseklik}cm", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if mevcut_yukseklik > 35:
                    cv2.putText(img, "H ARAMASI KAPALI - ONCE ZEMINE INILIYOR...", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    drone.send_rc_control(0, 0, dikey_hiz, 0)
                else:
                    h_x, h_y, h_alan, _ = mavi_h_tespit_et(img)

                    if h_x is not None:
                        drone.send_rc_control(0, 0, 0, 0)
                        print("[SİSTEM] Mavi H harfi görüldü! Doğrudan hizalanılıyor.")
                        mevcut_durum = DURUM_MAVI_H_HIZALAN
                        h_hizalanma_asama = 0
                        h_tarama_baslangic = 0
                    else:
                        if h_tarama_baslangic == 0:
                            h_tarama_baslangic = time.time()

                        gecen_sure = time.time() - h_tarama_baslangic

                        if gecen_sure < 3.0:
                            cv2.putText(img, "H TARAMASI: SAGA BAKIYOR", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 165, 255), 2)
                            drone.send_rc_control(0, 0, dikey_hiz, 35)
                        elif gecen_sure < 9.0:
                            cv2.putText(img, "H TARAMASI: SOLA BAKIYOR", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 165, 255), 2)
                            drone.send_rc_control(0, 0, dikey_hiz, -35)
                        elif gecen_sure < 12.0:
                            cv2.putText(img, "H TARAMASI: MERKEZE HİZALANIYOR", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 165, 255), 2)
                            drone.send_rc_control(0, 0, dikey_hiz, 35)
                        elif gecen_sure < 18.0:
                            cv2.putText(img, "H TARAMASI: KÖR NOKTA İÇİN YAVAŞÇA GERİ GİDİLİYOR...", (30, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            drone.send_rc_control(0, -15, dikey_hiz, 0)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            h_tarama_baslangic = time.time()

            elif mevcut_durum == DURUM_MAVI_H_HIZALAN:
                h_x, h_y, h_alan, h_kontur = mavi_h_tespit_et(img)
                mevcut_yukseklik = drone.get_height()

                dikey_hiz = max(-20, min(20, int((25 - mevcut_yukseklik) * 1.5)))

                if h_hizalanma_asama == 0:
                    if h_x is not None:
                        cv2.drawContours(img, [h_kontur], -1, (0, 255, 0), 3)
                        cv2.circle(img, (h_x, h_y), 7, (0, 0, 255), -1)
                        hata_x = h_x - frame_merkez_x
                        if abs(hata_x) > 25:
                            drone.send_rc_control(max(-20, min(20, int(hata_x / 4))), 0, dikey_hiz, 0)
                            cv2.putText(img, "ASAMA 1: YATAYDA ORTALANIYOR...", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 165, 255), 2)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            h_hizalanma_asama = 1
                    else:
                        drone.send_rc_control(0, 0, 0, 0)
                        if time.time() - h_tarama_baslangic > 2.0:
                            mevcut_durum = DURUM_MAVI_H_ARA
                            h_hizalanma_asama = 0
                            h_tarama_baslangic = 0

                elif h_hizalanma_asama == 1:
                    if h_x is not None:
                        cv2.drawContours(img, [h_kontur], -1, (0, 255, 0), 3)
                        cv2.circle(img, (h_x, h_y), 7, (0, 0, 255), -1)
                        sag_sol_hiz = max(-15, min(15, int((h_x - frame_merkez_x) / 5)))
                        if h_y < 650:
                            drone.send_rc_control(sag_sol_hiz, 20, dikey_hiz, 0)
                            cv2.putText(img, f"ASAMA 2: YAKLASILIYOR... (Y: {h_y}/650)", (30, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
                            h_ileri_baslangic = time.time()
                            h_hizalanma_asama = 2
                    else:
                        drone.send_rc_control(0, 0, 0, 0)
                        h_ileri_baslangic = time.time()
                        h_hizalanma_asama = 2

                elif h_hizalanma_asama == 2:
                    gecen_zaman = time.time() - h_ileri_baslangic
                    EKSTRA_ILERI_SURESI = 4.4
                    cv2.putText(img, f"ASAMA 3: KOR NOKTA MERKEZLEMESI ({gecen_zaman:.1f}/{EKSTRA_ILERI_SURESI}s)",
                                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if gecen_zaman < EKSTRA_ILERI_SURESI:
                        drone.send_rc_control(0, 20, dikey_hiz, 0)
                    else:
                        drone.send_rc_control(0, 0, 0, 0)
                        cv2.imshow("Otonom Parkur", img)
                        cv2.waitKey(1)
                        time.sleep(0.5)
                        mevcut_durum = DURUM_INIS

            elif mevcut_durum == DURUM_INIS:
                drone.land()
                break

            cv2.imshow("Otonom Parkur", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                drone.land()
                break

            elif cv2.waitKey(1) & 0xFF == ord('e'):
                drone.emergency()
                break

    except Exception as hata:
        print(f"\n[SİSTEM HATASI] {hata}")
        try:
            drone.land()
        except:
            pass

    finally:
        cv2.destroyAllWindows()
        try:
            drone.streamoff()
        except:
            pass


if __name__ == "__main__":
    main()
