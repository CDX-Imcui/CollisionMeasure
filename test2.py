from PIL import Image
print("Pillow version:", Image.__version__)
img = Image.open("image_00059.jpg")
img.show()
