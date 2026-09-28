from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import cv2
from pyzbar.pyzbar import decode
from sqlalchemy.exc import IntegrityError
from database import SessionLocal, ScanResult, Product

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("uploads", exist_ok=True)
os.makedirs("reference_images", exist_ok=True)


@app.get("/hello")
def say_hello():
    return {"message": "hi, I'm alive"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    save_path = os.path.join("uploads", file.filename)

    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    image = cv2.imread(save_path)

    if image is None:
        return {"error": "Could not read image — is it a valid image file?"}

    height, width, channels = image.shape

    return {
        "filename": file.filename,
        "status": "saved successfully",
        "width": width,
        "height": height,
        "channels": channels
    }


@app.post("/scan-barcode")
async def scan_barcode(file: UploadFile = File(...)):
    save_path = os.path.join("uploads", file.filename)

    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    image = cv2.imread(save_path)
    if image is None:
        return {"error": "Could not read image"}

    barcodes = decode(image)
    db = SessionLocal()

    if not barcodes:
        record = ScanResult(
            filename=file.filename,
            barcode_data=None,
            barcode_type=None,
            is_genuine=None
        )
        db.add(record)
        db.commit()
        db.close()
        return {"filename": file.filename, "barcodes_found": 0, "results": []}

    results = []
    for barcode in barcodes:
        data = barcode.data.decode("utf-8")
        btype = barcode.type

        match = db.query(Product).filter(Product.barcode == data).first()
        is_genuine = match is not None

        results.append({
            "data": data,
            "type": btype,
            "is_genuine": is_genuine,
            "product_name": match.name if match else None
        })

        record = ScanResult(
            filename=file.filename,
            barcode_data=data,
            barcode_type=btype,
            is_genuine=is_genuine
        )
        db.add(record)

    db.commit()
    db.close()

    return {
        "filename": file.filename,
        "barcodes_found": len(results),
        "results": results
    }


@app.get("/scans")
def get_all_scans():
    db = SessionLocal()
    scans = db.query(ScanResult).all()
    db.close()

    return [
        {
            "id": s.id,
            "filename": s.filename,
            "barcode_data": s.barcode_data,
            "barcode_type": s.barcode_type,
            "is_genuine": s.is_genuine,
            "scanned_at": s.scanned_at.isoformat()
        }
        for s in scans
    ]


@app.post("/products")
def add_product(barcode: str, name: str, manufacturer: str):
    db = SessionLocal()

    existing = db.query(Product).filter(Product.barcode == barcode).first()
    if existing:
        db.close()
        raise HTTPException(
            status_code=400,
            detail=f"A product with barcode {barcode} is already registered."
        )

    try:
        product = Product(barcode=barcode, name=name, manufacturer=manufacturer)
        db.add(product)
        db.commit()
        db.refresh(product)
    except IntegrityError:
        db.rollback()
        db.close()
        raise HTTPException(status_code=400, detail="Could not add product — barcode may already exist.")

    db.close()
    return {"id": product.id, "barcode": barcode, "name": name}


@app.post("/register-reference")
async def register_reference(barcode: str, file: UploadFile = File(...)):
    db = SessionLocal()
    product = db.query(Product).filter(Product.barcode == barcode).first()

    if not product:
        db.close()
        raise HTTPException(status_code=404, detail="No product found with this barcode. Register the product first.")

    save_path = os.path.join("reference_images", f"{barcode}.jpg")
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    db.close()
    return {"barcode": barcode, "status": "reference image saved", "path": save_path}


def compare_images(reference_path, uploaded_path):
    img1 = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(uploaded_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0, 0

    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    good_matches = [m for m in matches if m.distance < 50]

    total_keypoints = min(len(keypoints1), len(keypoints2))
    match_score = len(good_matches) / total_keypoints if total_keypoints > 0 else 0

    return len(good_matches), round(match_score * 100, 2)


@app.post("/verify-packaging")
async def verify_packaging(barcode: str, file: UploadFile = File(...)):
    reference_path = os.path.join("reference_images", f"{barcode}.jpg")

    if not os.path.exists(reference_path):
        raise HTTPException(status_code=404, detail="No reference image registered for this barcode yet.")

    uploaded_path = os.path.join("uploads", file.filename)
    with open(uploaded_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    good_matches, match_score = compare_images(reference_path, uploaded_path)

    verdict = "likely genuine" if match_score >= 30 else "likely counterfeit or unrelated product"

    return {
        "barcode": barcode,
        "good_matches": good_matches,
        "match_score_percent": match_score,
        "verdict": verdict
    }
@app.post("/debug-keypoints")
async def debug_keypoints(file: UploadFile = File(...)):
    save_path = os.path.join("uploads", "debug_" + file.filename)
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    img = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Could not read image"}

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)

    return {
        "filename": file.filename,
        "image_shape": img.shape,
        "keypoints_found": len(keypoints) if keypoints else 0,
        "has_descriptors": descriptors is not None
    }