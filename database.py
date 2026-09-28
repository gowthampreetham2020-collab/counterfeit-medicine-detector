from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///app.db")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    barcode_data = Column(String, nullable=True)
    barcode_type = Column(String, nullable=True)
    is_genuine = Column(Boolean, nullable=True)   # NEW: verdict for this scan
    scanned_at = Column(DateTime, default=datetime.utcnow)

# NEW TABLE: known genuine products
class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    barcode = Column(String, unique=True, index=True)
    name = Column(String)
    manufacturer = Column(String)

Base.metadata.create_all(bind=engine)