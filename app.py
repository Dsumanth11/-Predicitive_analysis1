# from crypt import methods
from flask import Flask,render_template,request

app=Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/whatpdis")
def hello1():
    return render_template("whatpdis.html")

@app.route("/Diseases")
def hello2():
    return render_template("Diseases.html")

@app.route("/feedback")
def hello3():
    return render_template("feedback.html")

@app.route("/Diabetes")
def hello4():
    return render_template("diabetes.html")

@app.route("/PCOS")
def hello5():
    return render_template("PCOS.html")

@app.route("/HeartAttack")
def hello6():
    return render_template("HeartAttack.html")

@app.route("/sub",methods=['POST'])
def submit():
    if request.method == "POST":
        a=request.form["pregnancy"]
        b=request.form["glucose"]
        c=request.form["bloodpressure"]
        d=request.form["SkinThickness"]
        e=request.form["insulin"]
        f=request.form["BMI"]
        g=request.form["DiabetesPedigreeFunction"]
        h=request.form["Age"]
        import pandas as pd
        import numpy as np
        df=pd.read_csv("diabetes.csv")

        x=df.iloc[:,0:8].values
        y=df.iloc[:,8].values

        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        from sklearn.linear_model import LogisticRegression
        Log= LogisticRegression(random_state=0)
        Log.fit(x_train,y_train)
        y_pred=Log.predict(x_test)
        z=Log.predict(sc.transform([[a,b,c,d,e,f,g,h]]))
        if z==1:
            name="YES"
        else:
            name="NO"
        
    return render_template("sub.html",n=name,a1=a,b1=b,c1=c,d1=d,e1=e,f1=f,g1=g,h1=h) 

@app.route("/HeartAttackResult",methods=['POST'])
def submit12():
    if request.method == "POST":
        age=request.form["age"]
        sex=request.form["sex"]
        cp=request.form["cp"]
        trtbps=request.form["trtbps"]
        chol=request.form["chol"]
        fbs=request.form["fbs"]
        restecg=request.form["restecg"]
        thalachh=request.form["thalachh"]
        exng=request.form["exng"]
        oldpeak=request.form["oldpeak"]
        slp=request.form["slp"]
        caa=request.form["caa"]
        thall=request.form["thall"]
        import pandas as pd
        import numpy as np
        df=pd.read_csv("heart.csv")
        x=df.iloc[:,0:13].values
        y=df.iloc[:,13].values
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        from sklearn.linear_model import LogisticRegression
        Log= LogisticRegression(random_state=0)
        Log.fit(x_train,y_train)
        y_pred=Log.predict(x_test)
        z=Log.predict(sc.transform([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]]))
        if z==1:
            name="YES"
        else:
            name="NO"
    return render_template("HeartAttackResult.html",n=name,age=age,sex=sex,cp=cp,trtbps=trtbps,chol=chol,fbs=fbs,restecg=restecg,thalachh=thalachh,exng=exng,oldpeak=oldpeak,slp=slp,caa=caa,thall=thall) 
    
@app.route("/PcosResult",methods=['POST'])
def submit1():
    if request.method == "POST":
        a1=request.form["Age"]
        a2=request.form["Weight"]
        a3=request.form["Height"]
        a4=request.form["BMI"]
        a5=request.form["Blood Group"]
        a6=request.form["Pulse rate"]
        a7=request.form["RR"]
        a8=request.form["Hb"]
        a9=request.form["Cycle"]
        a10=request.form["Cyclelength"]
        a11=request.form["MarraigeStatus"]
        a12=request.form["Pregnant"]
        a13=request.form["Noofaborptions"]
        a14=request.form["IbetaHCG"]
        a15=request.form["IIbetaHCG"]
        a16=request.form["FSH"]
        a17=request.form["LH"]
        a18=request.form["FSH"]
        a19=request.form["Hip"]
        a20=request.form["Waist"]
        a21=request.form["WaistHip"]
        a22=request.form["TSH"]
        a23=request.form["AMH"]
        a24=request.form["PRL"]
        a25=request.form["Vit"]
        a26=request.form["PRG"]
        a27=request.form["RBS"]
        a28=request.form["Weightgain"]
        a29=request.form["hairgrowth"]
        a30=request.form["Skindarkening"]
        a31=request.form["Hairloss"]
        a32=request.form["RegExercise"]
        a33=request.form["BPSystolic"]
        a34=request.form["BPDiastolic"]
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        df = pd.read_csv("PCOS_clean_data_without_infertility.csv")   
        df = df.rename(columns=lambda x: x.strip())
        df.corr()["PCOS (Y/N)"].sort_values(ascending=False)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print(y_pred)
        z=logreg.predict(sc.transform([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34]]))
        if z==1:
            name="YES"
        else:
            name="NO"
            
    return render_template("sub.html",n=name,a1=a1,a2=a2,a3=a3,a4=a4,a5=a5,a6=a6,a7=a7,a8=a8,a9=a9,a10=a10,a11=a11,a12=a12,a13=a13,a14=a14,a15=a15,a16=a16,a17=a17,a18=a18,a19=a19,a20=a20,a21=a21,a22=a22,a23=a23,a24=a24,a25=a25,a26=a26,a27=a27,a28=a28,a29=a29,a30=a30,a31=a31,a32=a32,a33=a33,a34=a34) 

if __name__=="__main__":
    app.run(debug=True)