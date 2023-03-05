from pydantic import BaseModel
# class with describes Bank note measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy:  float
