# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
from datetime import datetime
import paddle
import os

os.environ["FLAGS_use_mkldnn"] = "1"
os.environ["FLAGS_use_mkldnn_bfloat16"] = "0"
# paddle.set_device("cpu")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_model_name="PP-OCRv5_det",
    rec_model_name="PP-OCRv5_rec")

# Run OCR inference on a sample image
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- start")
result = ocr.predict(
    input="C:\\Users\\Jonas\\Downloads\\Test.png")
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- ende")
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- start")
result = ocr.predict(
    input="C:\\Users\\Jonas\\Downloads\\Test.png")
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- ende")
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- start")
result = ocr.predict(
    input="C:\\Users\\Jonas\\Downloads\\Test.png")
print (datetime.now().strftime("%H:%M:%S") + " ---------------------------- ende")

# Visualize the results and save the JSON results
for res in result:
    res.print()