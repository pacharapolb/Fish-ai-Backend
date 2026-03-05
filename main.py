from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# เปิดทางให้ Frontend จาก Vercel ยิงข้อมูลเข้ามาได้ (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # อนุญาตทุกเว็บยิงมาได้
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 โหลดโมเดล AI ของพี่ (เช็คชื่อไฟล์ให้ตรงกับที่อัปขึ้น GitHub นะครับ)
# ถ้าโมเดลพี่ชื่ออื่น ให้แก้ตรง "babyfish.pt" เป็นชื่อของพี่
try:
    model = YOLO("babyfish.pt") 
except Exception as e:
    print("พังง่ะ โหลดโมเดลไม่ได้:", e)
    model = None

@app.get("/")
def read_root():
    return {"status": "Backend พร้อมลุยแบบเฟี้ยวๆ!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"status": "error", "message": "เซิร์ฟเวอร์โหลดไฟล์โมเดล babyfish.pt ไม่สำเร็จ!"}

    try:
        # 1. อ่านไฟล์รูปภาพที่ส่งมาจากหน้าเว็บ
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. สั่งให้ YOLO ส่องหาปลาในรูป (ปรับ conf=0.25 คือความมั่นใจ 25% ขึ้นไปถึงจะนับ)
        results = model.predict(source=img, conf=0.1, iou=0.3, imgsz=320, half=True)

        # 3. นับจำนวนว่าเจอปลาทั้งหมดกี่ตัว
        count = len(results[0].boxes)

        # 4. ให้ AI วาดกรอบสี่เหลี่ยมทับลงไปบนรูป
        annotated_img = results[0].plot()

        # 5. แปลงรูปที่วาดกรอบแล้ว ให้เป็นข้อความรหัส Base64 
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 6. ส่งของขวัญกลับไปให้ Vercel (ข้อความ + รูปภาพ)
        return {
            "status": "success",
            "message": f"นับเสร็จแล้ว! ในรูป {file.filename} เจอลูกปลาทั้งหมด {count} ตัว!",
            "image_base64": img_base64  # << ตรงนี้แหละที่หน้าเว็บรอรับอยู่!
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"เกิดข้อผิดพลาดตอนคำนวณ: {str(e)}"
        }

