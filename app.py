import tkinter as tk
from tkinter import *
import os
import sqlite3
from functools import partial
from tkinter import Menu
import re
import smtplib, ssl
import random
from tkinter import messagebox

root=tk.Tk()
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
#bg img for login pg
#for full screen login page
root.geometry("{0}x{0}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.attributes('-zoomed', True)
#root bg color change 
root.configure(bg="gray25")
#giving title to root window
root.title("Mask detection Login")

#CREATING DATABASE OR CONNECTING 
logdb=sqlite3.connect("login.db")

#create cursor
c=logdb.cursor()
try:
    #create table
    c.execute("CREATE TABLE IF NOT EXISTS info(id INTEGER PRIMARY KEY,userid VARCHAR NOT NULL,password VARCHAR NOT NULL,email TEXT ,phone INTEGER);")
    dt=('1','end','end','end','end')
    c.execute("INSERT INTO info(id,userid,password,email,phone) VALUES(?,?,?,?,?);",dt)
    dt=('2','admin','admin123','admin@pcpoly.com','9436456456')
    c.execute("INSERT INTO info(id,userid,password,email,phone) VALUES(?,?,?,?,?);",dt)
except:
    print("DATA already exists in DataBase")
    #raise

#creating menus and sub menus
menubar = tk.Menu(root,bg="gray30",fg="snow")
filemenu = tk.Menu(menubar, tearoff=0,bg="gray30",fg="snow",font=("Antonio",16))
menubar.add_command(label="                                                                                                                                                                                                                                                                                                              ",activebackground="gray30")

#Label for bg img
l1=tk.Label(root,text="Mask Detection System \n LOGIN",fg="white",borderwidth=
4,relief="raised",font=("Antonio",40),bg="#5e3594")
l1.pack(side=TOP,fill=tk.X)

#user id label
user1 = tk.Label(root,text="  User  ID:",bg="gray25",fg="white",font=("Antonio",20))
user1.place(bordermode=OUTSIDE,relheight=0.038,relwidth=0.1,relx = 0.32, rely = 0.40)

#label for password
passw1 = tk.Label(root,text="Password:",bg="gray25",fg="white",font=("Antonio",20))
passw1.place(bordermode=OUTSIDE,relheight=0.038,relwidth=0.1,relx = 0.32, rely = 0.47)

#place(bordermode=OUTSIDE,relheight=0.075,relwidth=0.1,relx=0.8,rely=0.1)

#entry for user id
user = tk.Entry(root,bg="cyan",fg="Black",font=("Antonio",20))
user.place(bordermode=OUTSIDE,relheight=0.045,relwidth=0.15,relx=0.44, rely=0.40)

#entry for password
passw = tk.Entry(root,bg="cyan",fg="Black",show="*",font=("Antonio",20))
passw.place(bordermode=OUTSIDE,relheight=0.045,relwidth=0.15,relx=0.44, rely=0.47)

def ext():
    root.destroy()
def about():
    print("about called")
    #a1=Label(,text="HI We are P.C.P Students",bg="gray25",fg="white")
def contact():
    print("contact called")
    #c1=(,text="Email:G9maskdetectionsystem@gmail.com",bg="gray25",fg="white")

def onLeave(event):
    fp.config(fg='brown1')
 
def onEnter(event):
    fp.config(fg='white')
    
def forpas2(self):
    #import forpas
    window = tk.Toplevel(root)
    window.geometry('670x630')
    window.configure(bg="gray25")
    window.attributes('-zoomed', True)
    window.title("password recovery")

    regex = '(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'

    l1=tk.Label(window,text="Mask Detection System \n Password Recovery",fg="snow",relief="raised",borderwidth=4,font=("calibri",40),bg="#5e3594")
    l1.pack(side=TOP,fill=X)

    fpl = tk.Label(window,bg="gray60",text="Enter your registered email ID",relief="raised",fg="black",font=("times new roman",20))
    fpl.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.27,relx = 0.27, rely = 0.37)
    global fpe
    fpe = tk.Entry(window,bg="cyan2",fg="black",font=("times new roman",20))
    fpe.place(bordermode=OUTSIDE,relheight=0.037,relwidth=0.35,relx = 0.22, rely = 0.42)

    def goback1():
            window.destroy()


    def sendemail():
            # validate otp
            def verifyotp():
                            if lotp.get()==otp:
                                    c.execute("SELECT password FROM info DESC WHERE email=?",(fpedb,))
                                    records=c.fetchall()
                                    for row in records:
                                            passfor=row[0]
                                      # create smtp session 
                                    s = smtplib.SMTP("smtp.gmail.com" , 587)  # 587 is a port number
                                    # start TLS for E-mail security 
                                    s.starttls()
                                    # Auto Log in to your gmail account
                                    s.login("g9fmds09@gmail.com" , "G9FMDS@fmds09")
                                    #send email
                                    subject = 'Mask Detection System Password Recovery'
                                    body = 'This is Email for the Request of password recovery registered with this email id\n The password is :'

                                    msg = f'Subject: {subject}\n\n{body}\n{passfor}'
                                    s.sendmail("tejasharal003@gmail.com" , fpe.get(),msg)
                                    print("Email sent succesfully..")
                                    messagebox.showinfo("Recovery Email sent successfully","PASSWORD Recovery Email has been succesfully sent to\n"+fpedb.upper())
                                    fpe.delete(0,25)
                                    # close smtp session
                                    s.quit()
                                    root.destroy()
                                    import app
                    
            global otp
             # create smtp session 
            s = smtplib.SMTP("smtp.gmail.com" , 587)  # 587 is a port number
            # start TLS for E-mail security 
            s.starttls()
            # Auto Log in to your gmail account
            s.login("g9fmds09@gmail.com" , "G9FMDS@fmds09")
            otp = random.randint(1000, 9999)
            otp = str(otp)
                    
            s.sendmail("tejasharal003@gmail.com" , fpe.get(), otp)
            print("OTP sent succesfully..")
            # close smtp session
            s.quit()
            l4=Label(root,text="OTP sent successfully ! ",fg="red",font=("times new roman",18))
            l4.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.4,relx=0.25, rely=0.75)
            l4.after(3000, lambda: l4.destroy())

            window1 = tk.Toplevel(window)
            window1.geometry('550x370+800+170')
            window1.configure(bg="gray25")
            window1.title("OTP Verification")
            
            l1=tk.Label(window1,text="Mask Detection System \n OTP Verification",fg="snow",relief="raised",borderwidth=4,font=("calibri",40),bg="#5e3594")
            l1.pack(side=TOP,fill=X)
            
            loptl = tk.Label(window1,bg="gray60",text="Enter your OTP:",relief="raised",fg="black",font=("times new roman",20))
            loptl.place(bordermode=OUTSIDE,relheight=0.11,relwidth=0.35,relx = 0.37, rely = 0.42)

            lotp = tk.Entry(window1,bg="cyan2",fg="black",font=("times new roman",20))
            lotp.place(bordermode=OUTSIDE,relheight=0.11,relwidth=0.35,relx = 0.37, rely = 0.54)
            
            fpeo=tk.Button(window1,text=" Verify ",bg="#5e3594",fg="black",command=verifyotp,font=("times new roman",15))
            fpeo.place(relheight=0.09,relwidth=0.35,relx=0.37,rely=0.73)

            back1=tk.Button(window1,text=" GO Back To Login ",bg="#5e3594",fg="black",command=goback1,font=("times new roman",15))
            back1.place(relheight=0.09,relwidth=0.35,relx=0.37,rely=0.87)


    def pasrec():
            if not re.search(regex,fpe.get()):
                    l4=Label(root,text="Invalid email id",fg="red")
                    l4.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.15,relx=0.7, rely=0.42)
                    l4.after(3000, lambda: l4.destroy())
            elif re.search(regex,fpe.get()):
                    c.execute("SELECT * FROM info ORDER BY id DESC")
                    records=c.fetchall()
                    for row in records:
                            if row[3] != "end":
                                    global fpedb
                                    fpedb=row[3]
                                    if str(row[3])==fpe.get():#this one is a string:
                                            otpid=row[0]
                                            print("called")
                                            try:
                                                    sendemail()
                                                    break
                                            except:
                                                    raise
                                                    l4=Label(root,text="Some technical error Occured!!! \n Please Try Again",fg="red",font=("times new roman",18))
                                                    l4.place(bordermode=OUTSIDE,relheight=0.05,relwidth=0.4,relx=0.25, rely=0.75)
                                                    l4.after(2000, lambda: l4.destroy())
                                                    break
                                                    
                            else:
                                    print("called else")
                                    l4=Label(root,text="Email id not Registered! Try Another !",fg="red")
                                    l4.place(bordermode=OUTSIDE,relheight=0.04,relwidth=0.25,relx=0.7, rely=0.42)
                                    l4.after(2000, lambda: l4.destroy())
                                    fpe.delete(0,25)
                                    break
                            
    fpeb=tk.Button(window,text=" SUBMIT ",bg="#5e3594",fg="black",command=pasrec,font=("times new roman",15))
    fpeb.place(relheight=0.05,relwidth=0.2,relx=0.3,rely=0.5)

    back=tk.Button(window,text=" GO Back To Login ",bg="#5e3594",fg="black",command=goback1,font=("times new roman",15))
    back.place(relheight=0.05,relwidth=0.2,relx=0.3,rely=0.58)
        
fp = tk.Label(root,text="Forgot password?",bg="gray25",fg="brown1",font=("times new roman",18))
fp.place(bordermode=OUTSIDE,relheight=0.035,relwidth=0.15,relx = 0.3, rely = 0.55)
fp.bind("<Button-1>",forpas2)
fp.bind('<Leave>', onLeave)
fp.bind('<Enter>', onEnter)

fr=Frame(root,width=200,height=200,bg="gray25")
fr.place(relx=0.64,rely=0.43)

#function to validate login details using data base
def login():
    try:
        c.execute("SELECT * FROM info ORDER BY id DESC")
        records=c.fetchall()
        if len(user.get())==0 and len(passw.get())==0:
            l4a = tk.Label(fr,text="No Empty Fields Allowed!!",bg="gray25",fg="red",font=("Antonio",20))
            l4a.pack(padx=1,pady=1)
            l4a.after(3000, lambda: l4a.destroy())
        elif (len(user.get()) >= 5) and (len(passw.get()) >= 8):
                global flag
                flag=0
                for row in records:
                    if row[1]!='end':
                        if user.get()==row[1] and passw.get()==row[2]:
                            l2 = tk.Label(fr,text="Login Successfull",bg="gray25",fg="red",font=("Antonio",20))
                            l2.pack(padx=1,pady=1)
                            #destroy previous window after successfulll login
                            root.after(700, lambda: root.destroy())
                            import app2
                            flag=1
                        elif user.get()==row[1] and passw.get()!=row[2]:
                            l2 = tk.Label(fr,text="Wrong Password!!!",bg="gray25",fg="red",font=("Antonio",20))
                            l2.pack(padx=1,pady=1)
                            l2.after(3000, lambda: l2.destroy())
                            flag=1
                    elif flag==0:
                        l2 = tk.Label(fr,text="Wrong user ID!!! \n and Password",bg="gray25",fg="red",font=("Antonio",20))
                        l2.pack(padx=1,pady=1)
                        l2.after(3000, lambda: l2.destroy())
        else :
            l4 = tk.Label(fr,text="Please enter user id of 5 digits or more \n and password of 8 digits or more!!!",bg="gray25",fg="red",font=("times new roman",20))
            l4.pack(padx=1,pady=1)
            l4.after(3000, lambda: l4.destroy())
    except sqlite3.Error as error:
        l5 = tk.Label(fr,text="Failed to read data from records!!!",bg="gray25",fg="red",font=("Antonio",20))
        raise
        l5.pack(padx=1,pady=1)
        l5.after(3000, lambda: l5.destroy())


#login button for login function
pwm=tk.Button(root,text="LOGIN",padx=10,pady=5,fg="white",bg="dark green",bd=3,command=login,font=("Antonio",15))
pwm.place(relx=0.5,rely=0.55)

#menus in menu bar of login page
filemenu.add_command(label="ABOUT US", command=about)
filemenu.add_command(label="CONTACT US", command=contact)
filemenu.add_command(label="Exit",command=ext)
filemenu.add_separator
menubar.add_cascade(label="MENU", menu=filemenu)

root.config(menu=menubar)

root.mainloop()

logdb.commit()
c.close
logdb.close()
