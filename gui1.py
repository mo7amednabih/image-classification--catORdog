
import tkinter as tk  # create page
import tkinter.filedialog as tkFileDialog # upload and choose image or file
from PIL import Image, ImageTk # show image


status_text = "Prediction"

def open_image():
    global image  # تستخدم كلمة global للإشارة إلى أن المتغير image هو متغير عالمي يتم استخدامه داخل الدالة وخارجها.
    path = tkFileDialog.askopenfilename(filetypes=[("Image Files", ".jpg .png .gif")])  # يطلب من المستخدم اختيار ملف صورة من خلال نافذة اختيار الملف، ويحدد filetypes الامتدادات المقبولة للملفات (JPG، PNG، GIF).
    if path:  # إذا تم اختيار ملف الصورة بنجاح.
        im = Image.open(path)  # يفتح الملف الصورة في المسار المحدد.
        tkimage = ImageTk.PhotoImage(im)  # يحول الصورة إلى صورة متوافقة مع Tkinter.
        image = tkimage  # يحدث المتغير image ليحتوي على الصورة المحولة.
        image_label.config(image=tkimage)  # يحدث عنصر واجهة المستخدم image_label ليعرض الصورة المحولة.
        classifecation(path)  # يقوم بتنفيذ الدالة classification لتصنيف الصورة المحملة.



def train():
    return True


def classifecation(path):
    return True



# Create a tkinter window
window = tk.Tk()
window.title("Image Classification")
window.geometry("900x600")

# Create a frame for images
image_frame = tk.Frame(window)
image_frame.pack()

# Create a label for title
title_label = tk.Label(image_frame, text="We Use Pseudoinverse to check the pic cat or dog", font=("Platypi", 20, "bold"), padx=10, pady=10)
title_label.pack()


# Load the second image
image2 = Image.open("dog-or-cat.jpg")
photo2 = ImageTk.PhotoImage(image2)

# Create a label for the second image and position it in the image frame
image_label2 = tk.Label(image_frame, image=photo2)
image_label2.pack(side="left", padx=80, pady=10)

# Load the first image
image = Image.open("upload.jpg")
photo = ImageTk.PhotoImage(image)

# Create a label for the first image and position it in the image frame
image_label = tk.Label(image_frame, image=photo)
image_label.pack(side="left", padx=50, pady=10)

# Create a frame for the buttons and status label
button_frame = tk.Frame(window)
button_frame.pack()

# Create button to load and train the model
button_train = tk.Button(button_frame, text="Train Model", command=lambda: train(), background="lightblue", font=("Platypi", 20, "bold"))
button_train.grid(row=0, column=0, padx=150)

# Create button to select and classify image
button_select = tk.Button(button_frame, text="Upload Image", command=lambda: open_image(), background="lightblue", font=("Platypi", 20, "bold"))
button_select.grid(row=0, column=1, padx=100, pady=20)

# Create a label for status
status_label = tk.Label(button_frame, text=status_text, font=("Platypi", 20, "bold"), padx=10, pady=10, relief=tk.RAISED, borderwidth=2)
status_label.grid(row=1, columnspan=2)

# Start the tkinter event loop
window.mainloop()
