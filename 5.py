import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse

# Load OCR model (TrOCR)
def load_ocr_model():
    ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return ocr_processor, ocr_model

# Load Image Captioning model (BLIP)
def load_caption_model():
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return caption_processor, caption_model

# Perform OCR
def perform_ocr(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return ocr_text.strip()

# Perform Image Captioning
def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Combine both outputs
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Load models
    ocr_processor, ocr_model = load_ocr_model()
    caption_processor, caption_model = load_caption_model()

    # OCR
    print("\n🔍 Performing OCR...")
    ocr_text = perform_ocr(image, ocr_processor, ocr_model)

    # Captioning
    print("🖼️ Generating Caption...")
    caption = generate_caption(image, caption_processor, caption_model)

    # Final Output
    print("\n✅ Results:")
    print("📄 OCR Text:      ", ocr_text)
    print("📝 Image Caption: ", caption)

    return ocr_text, caption

# CLI Argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR + Image Captioning Script")
    parser.add_argument("image_path", type=str, help="sample1.jpg")
    args = parser.parse_args()
    
    process_image(args.image_path)