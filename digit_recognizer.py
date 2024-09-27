import io
from tkinter import *
from tkinter import colorchooser, ttk
from PIL import Image, ImageDraw, ImageGrab
import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
import xgboost as xgb
import pickle as pkl
import os


class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.pen_width = 13
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        # method for painting on the tkinter canvas
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.pen_width, fill=self.color_fg,
                               capstyle='round', smooth=True)
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None


    def clearcanvas(self):
        self.c.delete(ALL)

    def guess_the_number(self):
        # get the region of the canvas and take the screenshot of it
        x, y = self.c.winfo_rootx(), self.c.winfo_rooty()
        w, h = self.c.winfo_width() , self.c.winfo_height()
        pyautogui.screenshot('screenshot.png', region=(x, y, w, h))

        # crop the image for better model performance
        crop_image = Image.open('screenshot.png')
        width, height = crop_image.size
        left = 5
        top = height / 15
        right = 300
        bottom = height * 0.98
        crop_image = crop_image.crop((left, top, right, bottom))
        crop_image.save('screenshot1.png')

        # read the image and convert it to 28 x 28 np.array (then ravel it)
        im = cv2.imread('screenshot1.png', 0)
        img_pil = Image.fromarray(im)
        img_28x28 = np.array(img_pil.resize((28, 28)))
        img_28x28 = img_28x28.ravel()
        img_28x28 = img_28x28[np.newaxis, :]

        # load the model from the pickle file
        model = pkl.load(open('model_pickle.pkl', 'rb'))

        # make the prediction of the drawn picture
        prediction = model.predict_proba(img_28x28)

        pred_1 = Label(self.controls, text=f'Predicted number {np.argmax(prediction)}')
        pred_1.grid(row=2, column=1)
        pred_2 = Label(self.controls, text=f'Confidence level is {prediction[:, np.argmax(prediction)] * 100}')
        pred_2.grid(row=3, column=1)

        # remove locally created files after prediction
        os.remove('screenshot.png')
        os.remove('screenshot1.png')


    def drawWidgets(self):
        # set up the tkinter frame
        self.controls = Frame(self.master, padx=10, pady=5, width=10)
        
        # create buttons for canvas clearing and guessing the number
        button_clear = Button(self.controls, text='Clear the screen', command=self.clearcanvas)
        button_clear.grid(row=0, column=1)
        button_guess = Button(self.controls, text='Recognize the digit', command=self.guess_the_number)
        button_guess.grid(row=1, column=1)

        self.controls.pack(side='left')

        # set up the canvas
        self.c = Canvas(self.master, width=300, height=300, bg=self.color_bg)
        self.c.pack(fill=BOTH, expand=True)


# initialize Tk class and run it
win = Tk()
win.title("Digit recognizer")
win.resizable(0, 0)
main(win)
win.mainloop()

