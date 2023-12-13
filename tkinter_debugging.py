import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
image = Image.open('/home/kaarlahti/PycharmProjects/valuation_uncertainty/logo.png')
photo = ImageTk.PhotoImage(image)
label = tk.Label(root, image=photo)
label.pack()
root.mainloop()
