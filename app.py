from flask import Flask, render_template, Response

from cam01_faceRec import faceRec
from cam02_faceRec import faceRec2
from cam03_faceRec import faceRec3

app = Flask(__name__)

def getVideo():
    faceRec()
    faceRec2()
    faceRec3()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/cam1')
def cam1():
    return render_template('cam1.html')
@app.route('/cam2')
def cam2():
    return render_template('cam2.html')
@app.route('/cam3')
def cam3():
    return render_template('cam3.html')

@app.route('/video1')
def videoshow():
    return Response(faceRec(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video2')
def videoshow2():
    return Response(faceRec2(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video3')
def videoshow3():
    return Response(faceRec3(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/list')
def list():
    return "<h1> Data list </h1>"

if __name__ == "__main__":
    app.run(debug=True)
