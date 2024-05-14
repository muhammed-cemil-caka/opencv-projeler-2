import cv2
import mediapipe as mp

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

kamera = cv2.VideoCapture(0)

toplam_parmak_sayisi_tum_eller = 0
numara = 0

with mp_eller.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as eller: 
    while kamera.isOpened():
        ret, cerceve = kamera.read()

        cerceve = cv2.flip(cerceve, 1)
        
        resim = cv2.cvtColor(cerceve, cv2.COLOR_BGR2RGB)
        
        sonuclar = eller.process(resim)
        
        resim = cv2.cvtColor(resim, cv2.COLOR_RGB2BGR)
        
        if sonuclar.multi_hand_landmarks:
            toplam_parmak_sayisi_tum_eller = 0
            for numara, el in enumerate(sonuclar.multi_hand_landmarks):
                mp_cizim.draw_landmarks(resim, el, mp_eller.HAND_CONNECTIONS, 
                                        mp_cizim.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_cizim.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                basparmak = el.landmark[mp_eller.HandLandmark.THUMB_TIP]
                
                isaret = el.landmark[mp_eller.HandLandmark.INDEX_FINGER_TIP]
                orta = el.landmark[mp_eller.HandLandmark.MIDDLE_FINGER_TIP]
                yuzuk = el.landmark[mp_eller.HandLandmark.RING_FINGER_TIP]
                serce = el.landmark[mp_eller.HandLandmark.PINKY_TIP]
                
                parmaklar = [isaret, orta, yuzuk, serce]
                toplam_parmaklar = 0
                
                if basparmak.x < isaret.x:
                    toplam_parmaklar += 1
                
                for i in range(0, len(parmaklar) - 1):
                    if parmaklar[i].y < parmaklar[i - 1].y:
                        toplam_parmaklar += 1
                
                toplam_parmak_sayisi_tum_eller += toplam_parmaklar
                
                cv2.putText(resim, f'El {numara+1}: {toplam_parmaklar}', (20, 50 + numara*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.putText(resim, f'Toplam Parmak Sayisi: {toplam_parmak_sayisi_tum_eller}', (20, 50 + (numara+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('El İzleme', resim)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

kamera.release()
cv2.destroyAllWindows()
