from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# เปิดทางให้ Frontend จาก Vercel ยิงข้อมูลเข้ามาได้ (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # อนุญาตทุกเว็บยิงมาได้
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Backend พร้อมลุยแบบเฟี้ยวๆ!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 📌 ตรงนี้คือจุดที่คุณเอาไปต่อกับโค้ด AI ของคุณ (เช่น โหลด YOLO มานับปลา/คัดถั่ว)
    # ตอนนี้เราจำลองผลลัพธ์ไปก่อน
    
    file_name = file.filename
    # จำลองว่า AI นับเสร็จแล้ว
    fake_result = f"นับเสร็จแล้ว! เจอของดีในรูป {file_name} เต็มไปหมด!"
    
    return {
        "status": "success",
        "message": fake_result
    }

# 💡 ตอนรันจริงใน Render อย่าลืมสร้างไฟล์ requirements.txt
# ข้างในเขียน: fastapi uvicorn python-multipart