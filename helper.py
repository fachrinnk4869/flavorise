
from together import Together
from dotenv import load_dotenv
import base64
import mimetypes
load_dotenv()


def image_to_data_url(image_path):
    """
    Membaca file gambar, mengonversinya ke Base64, dan mengembalikannya sebagai Data URL.
    """
    # Tebak tipe MIME dari file gambar (misalnya, 'image/jpeg', 'image/png')
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Jika tidak bisa ditebak, gunakan default 'application/octet-stream'
        mime_type = "application/octet-stream"

    # Buka file dalam mode baca biner ('rb')
    with open(image_path, "rb") as image_file:
        # Baca konten biner file
        binary_data = image_file.read()
        # Encode konten biner ke Base64 dan ubah menjadi string
        base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
        # Kembalikan sebagai Data URL yang diformat dengan benar
        return f"data:{mime_type};base64,{base64_encoded_data}"


class MultimodalModel:
    def __init__(self):
        self.client = Together()

    def generate(self, image_path):
        # Ubah path gambar menjadi Data URL
        data_url = image_to_data_url(image_path)
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Apa saja yang ada di gambar itu? sebutkan saja bahan makanan nya, dipisah oleh koma. Jangan tambahkan penjelasan apapun selain daftar bahan makanannya."},
                    {"type": "image_url", "image_url": {
                        "url": f"{data_url}"}}
                ]
            }]
        )
        return response.choices[0].message.content
