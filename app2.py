import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog,filedialog
from tkinter import Menu
import os
import sqlite3
import cv2
import re
from tkinter import Tk, filedialog, Frame, Button, Canvas
from PIL import Image, ImageTk
import numpy as np

root=tk.Tk()

#bg img for login pg

#for full screen login page

w = root.winfo_screenwidth()
h = root.winfo_screenheight()

root.geometry("%dx%d+0+0" % (w, h))
root.attributes('-zoomed', True)

#giving title to root window
root.title("Mask detection System")

#change background of root
root.configure(bg="grey25")

#connecting to old database
logdb=sqlite3.connect("login.db")

#create cursor
c=logdb.cursor()

mask3 = Image. open('imgs/mask3.jpeg')
# The (450, 350) is (height, width)
mask3 = mask3. resize((750,750), Image. ANTIALIAS)
mask3 = ImageTk. PhotoImage(mask3)

#Label for bg text
l1=tk.Label(root,text="  Mask Detection System  ",fg="white",font=("Antonio",45),bg="#5e3594",borderwidth=4,relief="raised") 
l1.pack(side=TOP,fill=tk.X,ipadx=10,ipady=10,)
##l1=tk.Label(root,text="Logged in as :",fg="white",font=("Antonio",35),bg="#5e3594",borderwidth=4,relief="raised") 
##l1.pack(side=TOP,fill=tk.X)

l1=tk.Label(root,image=mask3,fg="white",font=("Antonio",45),bg="#5e3594",borderwidth=4,relief="raised") 
l1.pack(side=TOP,fill=tk.X,ipadx=10,ipady=10,)


#creating menus and sub menus
menubar = tk.Menu(root,bg="gray30",fg="snow")
filemenu = tk.Menu(menubar, tearoff=0,bg="gray30",fg="snow",font=("Antonio",18))
labelmenuwidth="__________________________________________"
labelmenuwidth*=4
menubar.add_command(label=" "+labelmenuwidth,activebackground="gray30")
#add spaces from col no. 27 till col no.329 i.e 302 spaces for menu
#menus options added in line 208

#l1.place(relheight=0.08,relwidth=0.45,relx=0.3,rely=0.01)
##l3=Label(root,text="WELCOME",fg="white",font=("Antonio",25),bg="grey25")
##l3.place(relheight=0.08,relwidth=0.18,relx=0.35,rely=0.07)
##l3.after(5000,lambda: l3.destroy())
l2=Label(root,text="PRESS ' Q ' TO Stop Stream ",fg="white",font=("Antonio",28),bg="grey25")
l2.place(relheight=0.08,relwidth=0.35,relx=0.325,rely=0.14)

regex = '(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'

#function to register a admin
def regtr(self):
    global name,pwd,pwd2,emal,phn,did
    c.execute("SELECT id FROM info ORDER BY id DESC")
    records=c.fetchall()
    for row in records:
        if row!='end':
            did=row[0]
            break
    def facereg1():
        cam = cv2.VideoCapture(0)
       # cam = cv2.VideoCapture('')
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier('facerecog/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id
        face_id = did

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while(True):

            ret, img = cam.read()
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("facerecog/dataset/admin/User." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('Face Capture \'press ESC to close\'', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                 break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def reg1():
        try:
            #print("reg1 entered")
            if name.get()=="" and pwd.get()=="":
                l4=Label(window,text="*This feild is required",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.32)
                l4.after(3000, lambda: l4.destroy())
                l41=Label(window,text="*This feild is required",fg="red")
                l41.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l41.after(3000, lambda: l41.destroy())
            elif len(pwd.get())<8 and len(name.get())<5:
                l4=Label(window,text="Password length above 8\n and username above 5! ",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l4.after(3000, lambda: l4.destroy())
                name.delete(0,15)
                pwd.delete(0,15)
            elif pwd.get()!=pwd2.get():
                l4=Label(window,text="Passwords not matching!",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l4.after(3000, lambda: l4.destroy())
                pwd.delete(0,15)
                pwd2.delete(0,15)
            elif not re.search(regex,emal.get()):
                l4=Label(window,text="Invalid email id",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.56)
                l4.after(3000, lambda: l4.destroy())
                emal.delete(0,15)
            elif len(phn.get())!=10:
                l4=Label(window,text="Plz enter mobile no. of 10 DIGITS",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.28,relx=0.65, rely=0.64)
                l4.after(3000, lambda: l4.destroy())
                phn.delete(0,15)
            else:
                #insert in database
                data=(did,name.get(),pwd.get(),emal.get(),phn.get())
                #query to enter in database
                c.execute("INSERT INTO info(id,userid,password,email,phone) VALUES(?,?,?,?,?);",data)
                logdb.commit()
                #c.execute("SELECT *FROM info")
##                l4=Label(window,text="Registration of "+name.get()+" is successfull ",fg="green1",font=("times new roman",16))
##                l4.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.7,relx=0.35, rely=0.85)
##                l4.after(3000, lambda: l4.destroy())
                messagebox.showinfo("Registration Successfull", "REGISTRATION OF "+name.get()+" IS SUCCESSFULLY DONE")
                #d.delete(0,15)
                name.delete(0,15)
                pwd.delete(0,15)
                pwd2.delete(0,15)
                emal.delete(0,15)
                phn.delete(0,15)
                #d=c.fetchall()
                #print("\n",d)
                window.after(1000, lambda: window.destroy())
                print("query successful")
        except sqlite3.Error as er:
            print(er)
        #REG1 functions end here

    window = tk.Toplevel(root)
    window.geometry('670x630')
    window.configure(bg="gray25")
    #Label for reg popup text
    l1=Label(window,text="  Mask Detection System  ",fg="white",bg="#5e3594",relief="raised",font=("calibri",28))
    l1.pack(side=TOP,fill=X,ipady=15)
    l3=Label(window,text="!REGISTER!",fg="white",bg="gray25",font=("calibri",24))
    l3.pack(side=TOP)
    l2=Label(window,text=" NEW ADMIN USER",fg="white",bg="gray25",font=("calibri",18))
    l2.pack(side=TOP)

    #labels for entries
    
    ld  = tk.Label(window,text="system gen. ID.:",fg="white",bg="gray25",font=("calibri",16))
    lname = tk.Label(window,text="* User Name:",fg="white",bg="gray25",font=("calibri",16))
    lpwd  = tk.Label(window,text="* Password:",fg="white",bg="gray25",font=("calibri",16))
    lpwd2 = tk.Label(window,text="verify Password:",fg="white",bg="gray25",font=("calibri",16))
    lemal = tk.Label(window,text="Email:",fg="white",bg="gray25",font=("calibri",16))
    lphn  = tk.Label(window,text="Phone No.:",fg="white",bg="gray25",font=("calibri",16))

    #set labels
    
    ld.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.15, rely=0.26)
    lname.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.17, rely=0.32)
    lpwd.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.18, rely=0.4)
    lpwd2.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.26,relx=0.14, rely=0.48)
    lemal.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.21, rely=0.56)
    lphn.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.18, rely=0.64)
    

    # Make entries

    did+=1
    d = tk.Label(window,text=did,bg="cyan",fg="black",font=("calibri",14))
    #d.insert(END, '%s'%did)
    name = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))
    pwd = tk.Entry(window,bg="cyan",fg="black",show="*",font=("calibri",14))
    pwd2 = tk.Entry(window,bg="cyan",fg="black",show="*",font=("calibri",14))
    emal = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))
    phn = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))

    #set entries
    
    d.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.26)
    name.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.32)
    pwd.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.4)
    pwd2.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.48)
    emal.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.56)
    phn.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.64)

    #move back to app2
    def ext():
        window.destroy()
        
    #button
    regtrb=tk.Button(window,text=" REGISTER ",bg="#b066e8",fg="snow",command=reg1).place(relheight=0.06,relwidth=0.23,relx=0.43,rely=0.74)
    #regface=tk.Button(window,text=" REGISTER FACE ID ",bg="#b066e8",fg="snow",command=facereg1).place(relheight=0.06,relwidth=0.17,relx=0.73,rely=0.74)
    ext=Button(window,text="Back",fg="white",command=ext,bg="#b066e8")
    ext.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.14,relx=0.05,rely=0.74)
    #function to register a admin


try:
    c.execute("CREATE TABLE IF NOT EXISTS usrinfo(id INTEGER PRIMARY KEY,userid VARCHAR NOT NULL,enrol INTEGER NOT NULL,password VARCHAR NOT NULL,email TEXT ,phone INTEGER);")
    dt=('1','end','end','end','end','end')
    c.execute("INSERT INTO usrinfo(id,userid,enrol,password,email,phone) VALUES(?,?,?,?,?,?);",dt)
    dt=('2','admin','1816460095','admin123','admin@pcpoly.com','9436456456')
    c.execute("INSERT INTO usrinfo(id,userid,enrol,password,email,phone) VALUES(?,?,?,?,?,?);",dt)
except:
    print("data exists")

def regtruser(self):
    global name,pwd,pwd2,emal,phn,died
    c.execute("SELECT id FROM usrinfo ORDER BY id DESC")
    records=c.fetchall()
    for row in records:
        if row!='end':
            died=row[0]
            break
    def facereg1():
        #cam = cv2.VideoCapture(0)
        cam = cv2.VideoCapture('http://192.168.31.81:4747/video')
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier('facerecog/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id
        face_id = died

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while(True):

            ret, img = cam.read()
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("facerecog/dataset/usr/User." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('Face Capture \'press ESC to close\'', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                 break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    def reg1():
        try:
            #print("reg1 entered")
            if name.get()=="" and pwd.get()=="":
                l4=Label(window,text="*This feild is required",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.32)
                l4.after(3000, lambda: l4.destroy())
                l41=Label(window,text="*This feild is required",fg="red")
                l41.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l41.after(3000, lambda: l41.destroy())
            elif len(pwd.get())<8 and len(name.get())<5:
                l4=Label(window,text="Password length above 8\n and username above 5! ",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l4.after(3000, lambda: l4.destroy())
                name.delete(0,15)
                pwd.delete(0,15)
            elif pwd.get()!=pwd2.get():
                l4=Label(window,text="Passwords not matching!",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.4)
                l4.after(3000, lambda: l4.destroy())
                pwd.delete(0,15)
                pwd2.delete(0,15)
            elif not re.search(regex,emal.get()):
                l4=Label(window,text="Invalid email id",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.25,relx=0.75, rely=0.56)
                l4.after(3000, lambda: l4.destroy())
                emal.delete(0,15)
            elif len(phn.get())!=10:
                l4=Label(window,text="Plz enter mobile no. of 10 DIGITS",fg="red")
                l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.28,relx=0.65, rely=0.64)
                l4.after(3000, lambda: l4.destroy())
                phn.delete(0,15)
            else:
                #insert in database
                data=(died,name.get(),enrol.get(),pwd.get(),emal.get(),phn.get())
                
                #query to enter in database
                c.execute("INSERT INTO usrinfo(id,userid,enrol,password,email,phone) VALUES(?,?,?,?,?,?);",data)
                logdb.commit()
                #c.execute("SELECT *FROM info")
##                l4=Label(window,text="Registration of "+name.get()+" is successfull ",fg="green1",font=("times new roman",16))
##                l4.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.7,relx=0.35, rely=0.85)
##                l4.after(3000, lambda: l4.destroy())
                messagebox.showinfo("Registration Successfull", "REGISTRATION OF USER "+name.get()+" IS SUCCESSFULLY DONE")
                #d.delete(0,15)
                name.delete(0,15)
                pwd.delete(0,15)
                pwd2.delete(0,15)
                emal.delete(0,15)
                phn.delete(0,15)
                enrol.delete(0,15)
                #d=c.fetchall()
                #print("\n",d)
                window.after(1000, lambda: window.destroy())
                print("query successful")
        except sqlite3.Error as er:
            print(er)
        #REG1 functions end here

    window = tk.Toplevel(root)
    window.geometry('670x630')
    window.configure(bg="gray25")
    #Label for reg popup text
    l1=Label(window,text="  Mask Detection System  ",fg="white",bg="#5e3594",relief="raised",font=("calibri",28))
    l1.pack(side=TOP,fill=X,ipady=15)
    l3=Label(window,text="!REGISTER!",fg="white",bg="gray25",font=("calibri",24))
    l3.pack(side=TOP)
    l2=Label(window,text=" NEW CLIENT USER",fg="white",bg="gray25",font=("calibri",18))
    l2.pack(side=TOP)

    #labels for entries
    
    ld  = tk.Label(window,text="system gen. ID.:",fg="white",bg="gray25",font=("calibri",16))
    lname = tk.Label(window,text="* User Name:",fg="white",bg="gray25",font=("calibri",16))
    lenrol = tk.Label(window,text="*Enrollment No:",fg="white",bg="gray25",font=("calibri",16))
    lpwd  = tk.Label(window,text="* Password:",fg="white",bg="gray25",font=("calibri",16))
    lpwd2 = tk.Label(window,text="verify Password:",fg="white",bg="gray25",font=("calibri",16))
    lemal = tk.Label(window,text="Email:",fg="white",bg="gray25",font=("calibri",16))
    lphn  = tk.Label(window,text="Phone No.:",fg="white",bg="gray25",font=("calibri",16))

    #set labels
    
    ld.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.15, rely=0.26)
    lname.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.17, rely=0.32)
    lenrol.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.17, rely=0.4)
    lpwd.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.18, rely=0.48)
    lpwd2.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.26,relx=0.14, rely=0.56)
    lemal.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.21, rely=0.64)
    lphn.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.18, rely=0.72)
    

    # Make entries

    died+=1
    d = tk.Label(window,text=died,bg="cyan",fg="black",font=("calibri",14))
    #d.insert(END, '%s'%did)
    name = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))
    enrol = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))
    pwd = tk.Entry(window,bg="cyan",fg="black",show="*",font=("calibri",14))
    pwd2 = tk.Entry(window,bg="cyan",fg="black",show="*",font=("calibri",14))
    emal = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))
    phn = tk.Entry(window,bg="cyan",fg="black",font=("calibri",14))

    #set entries
    
    d.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.26)
    name.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.32)
    enrol.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.4)
    pwd.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.48)
    pwd2.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.56)
    emal.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.64)
    phn.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.23,relx=0.43, rely=0.72)

    #move back to app2
    def ext():
        window.destroy()
        
    #button
    regtrb=tk.Button(window,text=" REGISTER ",bg="#b066e8",fg="snow",command=reg1).place(relheight=0.06,relwidth=0.23,relx=0.43,rely=0.82)
    regface=tk.Button(window,text=" REGISTER FACE ID",bg="#b066e8",fg="snow",command=facereg1).place(relheight=0.06,relwidth=0.2,relx=0.73,rely=0.82)
    ext=Button(window,text="Back",fg="white",command=ext,bg="#b066e8")
    ext.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.14,relx=0.05,rely=0.82)

#creating func for about info
    
def onLeave(event):
        lfr41.config(bg='gray40')
        
def onEnter(event):
        lfr41.config(bg='red')

def onLeave2(event):
        lfr42.config(bg='gray40')
        
def onEnter2(event):
        lfr42.config(bg='red')


def trainface():
    try:
        print("REFRESHING>.......")
        # Path for face image database
        path = 'facerecog/dataset/usr'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("facerecog/haarcascade_frontalface_default.xml");

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)

            return faceSamples,ids

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('facerecog/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        print("REFRESHED>.........")
    except:
        print("NO DATA TO REFRESH")

def showinfo():
    windowshow = tk.Toplevel(root)
    windowshow.geometry('800x800')
    windowshow.configure(bg="gray25")

    def select_image(): 
            file_path = filedialog.askopenfilename( initialdir="pwom",)
            des = Image.open(file_path)
            bg_image = ImageTk.PhotoImage(des)
            canvas.bg_image = bg_image
            canvas.create_image(330,300, image=canvas.bg_image)
            
    select = Button(windowshow, text="select an image", command=select_image,bg="green",fg="black",font=("times new roman",17))
    select.pack()
    canvas = Canvas(windowshow, width= 650, height=550, bg="grey")
    canvas.pack()
            
    windowshow.mainloop()

def about():
    print("about called")
    #a1=Label(,text="HI We are P.C.P Students",bg="gray25",fg="white")

def contact():
    print("contact called")
    #c1=(,text="Email:",bg="gray25",fg="white")

def logout():
    d = messagebox.askquestion("Logout","Are You sure to log out!!")
    if (d == 'yes'):
        messagebox.showinfo("Logout","YOU Are Logged out")
        root.after(300, lambda: root.destroy())
        import app

#Frame to show Notifications
fr4=Frame(root,width=380,height=750,bg="gray60",relief="sunken",borderwidth=5)
fr4.place(relx=0.7,rely=0.125)


mask1 = Image. open('imgs/mask1.png')
mask1 = mask1. resize((330,330), Image. ANTIALIAS)
mask1 = ImageTk. PhotoImage(mask1)

lfr41=Label(fr4,fg="brown1",image=mask1,font=("Antonio",14),bg="gray45",relief = "raised")
lfr41.place(relheight=0.5,relwidth=1,relx=0.0,rely=0.0)
lfr41.bind("<Button-1>",regtr)
lfr41.bind('<Leave>', onLeave)
lfr41.bind('<Enter>', onEnter)

mask2 = Image. open('imgs/mask2.jpg')
mask2 = mask2. resize((330,330), Image. ANTIALIAS)
mask2 = ImageTk. PhotoImage(mask2)

lfr42=Label(fr4,fg="brown1",image=mask2,font=("Antonio",14),bg="gray45",relief = "raised")
lfr42.place(relheight=0.5,relwidth=1,relx=0.0,rely=0.5)
lfr42.bind("<Button-1>",regtruser)
lfr42.bind('<Leave>', onLeave2)
lfr42.bind('<Enter>', onEnter2)


##lfr43=Label(fr4,text="Error",fg="brown1",font=("Antonio",14),bg="gray45",relief = "raised")
##lfr43.place(relheight=0.25,relwidth=1,relx=0.0,rely=0.5)
##lfr43.bind("<Button-1>",showinfo)
##lfr43.bind('<Leave>', onLeave3)
##lfr43.bind('<Enter>', onEnter3)
##
##
##lfr44=Label(fr4,text="Error",fg="brown1",font=("Antonio",14),bg="gray45",relief = "raised")
##lfr44.place(relheight=0.25,relwidth=1,relx=0.0,rely=0.75)
##lfr44.bind("<Button-1>",showinfo)
##lfr44.bind('<Leave>', onLeave4)
##lfr44.bind('<Enter>', onEnter4)



def startstream():
    import detect_mask_video
    
##def enterstream():
##    root.after(300, lambda: root.destroy())
     
def stopstream():
    root.after(300, lambda: root.destroy())

fr2=Frame(root,width=380,height=750,bg="gray40",relief="sunken",borderwidth=4)
fr2.place(relx=0.001,rely=0.125)

# Create Buttons
##reg = Button(root,text="REGISTER",command=regtr,fg="white",bg="#b066e8")
##logout = Button(root,text="Logout",command=logout,fg="white",bg="#b066e8")
sls=Button(fr2,text=" START LIVE STREAM",fg="white",command=startstream,bg="#b066e8")
pwom=Button(fr2,text="Browse Images",fg="white",command=showinfo,bg="#b066e8")
refresh=Button(fr2,text="Refresh Database",fg="white",command=trainface,bg="#b066e8")
#refreshadmin=Button(fr2,text="Refresh USER Database",fg="white",command=trainfaceusr,bg="#b066e8")
#ls=Button(root,text=" ENTER LIVE STREAM",command=enterstream,bg="#b066e8")
sts=Button(fr2,text=" STOP STREAM AND QUIT APP",fg="white",command=stopstream,bg="#b066e8")


#reg1=Label(root,text="Register \na new user  ⇩⇩!",bg="snow",fg="red3",font=("times new roman",12))

#setting options on menu

##filemenu.add_command(label="REGISTER", command=regtr)
##filemenu.add_command(label="REGISTER USER", command=regtruser)
filemenu.add_command(label="ABOUT US", command=about)
filemenu.add_command(label="CONTACT US", command=contact)
filemenu.add_command(label="LOG-OUT", command=logout)
filemenu.add_separator
menubar.add_cascade(label="MENU", menu=filemenu)

#set pos

##reg.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.1,relx=0.9,rely=0.17)
##logout.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.1,relx=0.9,rely=0.9)
#reg1.place(bordermode=OUTSIDE,relheight=0.08,relwidth=0.12,relx=0.8,rely=0.085)
sls.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.7,relx=0.08,rely=0.17)
pwom.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.7,relx=0.08,rely=0.27)
refresh.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.7,relx=0.08,rely=0.37)
#refreshadmin.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.7,relx=0.08,rely=0.47)
#ls.place(bordermode=OUTSIDE,relheight=0.055,relwidth=0.2,relx=0.0,rely=0.45)
sts.place(bordermode=OUTSIDE,relheight=0.06,relwidth=0.7,relx=0.08,rely=0.57)

#configuiring menubar on root
root.config(menu=menubar)

root.mainloop()
#c.execute("INSERT INTO info(userid,password,email,phone) VALUES('end','end','end','end');")
logdb.commit()
c.close
logdb.close()
