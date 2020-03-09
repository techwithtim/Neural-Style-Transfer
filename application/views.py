from flask import Blueprint, current_app
from flask import Flask, render_template, url_for, redirect, request, session, jsonify, flash, Blueprint
from werkzeug.utils import secure_filename
from .database import DataBase
from .style_transfer import Model, Settings
import os
import _thread
import pickle
import PIL.Image
import gevent
from flask import copy_current_request_context

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = "application/static/images/"
CONTENT_FOLDER = BASE_DIR + "content/"
STYLE_FOLDER = BASE_DIR + "style/"
DEFAULT =  "images/default.png"

view = Blueprint("views", __name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@view.route("/", methods=["POST","GET"])
def default():
    return redirect(url_for("views.home"))

@view.route("/home", methods=["POST","GET"])
def home():
    style_file = DEFAULT
    content_file = DEFAULT

    if request.method == "POST":
        if len(request.form) > 0:
            # HANDLE POST FOR GENERATE BUTTON CLICK
            valid = True
            settings = Settings()

            # check we have valid images
            if "style_file" in session and "content_file" in session:
                style_filename = session["style_file"].split("/")[-1]
                content_filename = session["content_file"].split("/")[-1]

                settings.set_style_image(style_filename)
                settings.set_content_image(content_filename)
            else:
                valid = False
                flash("RPlease upload a content and style image.")
                

            keys = ["variance", "style", "content", "steps"]
            values = []
            for key in keys:
                if key not in request.form:
                    if valid:
                        flash("RInvalid form. Please try again.")
                    valid = False
                    break
                values.append(request.form[key])

            if valid:
                variance, style, content, steps = values
                try:
                    settings.set_variation(int(variance))
                    settings.set_content_weight(float(content[:-1])/100)
                    settings.set_style_weight(float(style[:-1])/100)
                    model = Model(settings)
                    flash("GModel has started training.")

                    @copy_current_request_context
                    def work():
                        session["output"] = model.generate()
                        session["step"] = 1
                        flash("Step: " + str(session["step"]))
            
                    gevent.spawn(work)
                    
                except Exception as e:
                    flash("R [EXCEPTION]" + str(e))
                
            return redirect(request.url)
        else:
            # HANDLE POST FOR FILE UPLAOD
            keys = []
            if 'content_file' in request.files:
                folder = CONTENT_FOLDER
                keys.append(["content_file",folder])
            if "style_file" in request.files:
                folder = STYLE_FOLDER
                keys.append(["style_file", folder])
                
            if not keys:  
                return redirect(request.url)
            
            for k in keys:
                key, folder = k
                file = request.files[key]
                if file.filename == "":
                    continue
                
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(folder, filename))
                    session[key] = filename
    
    data = {"step": None, "content_image": content_file, "style_image": style_file, "output_image": DEFAULT}
    
    if "step" in session:
        data["step"] = session["step"]

    if "style_file" in session:
        data["style_image"] = "images/style/" + session["style_file"]
    
    if "content_file" in session:
        data["content_image"] = "images/content/" + session["content_file"]

    return render_template("home.html", **data)

@view.route("/history")
def history():
    return render_template("history.html")


@view.route("/get_default_image")
def get_default_image():
    """
    :return: a json object storing name of default image path
    """
    data = {"image-name": DEFAULT}
    return jsonify(data)


@view.route("/reset")
@view.route("/clear")
def clear():
    if "style_file" in session:
        session.pop("style_file")
    if "content_file" in session:
        session.pop("content_file")
    if "step" in session:
        session.pop("step")
    
    return redirect(url_for("views.home"))


@view.route("/flash_message")
def flash_message(): 
    """
    flashes a coloured message on the screen
    Options:
    R - red
    G - green
    """
    message = request.args.get("message")
    color = request.args.get("color")

    if color and message:
        flash(color+message)

    return jsonify({})

@view.route("/update")
def update():
    print(request.json)
    if "step" in request.json:
        session["step"] = request.json.get("step")
        print(session["step"])

    return jsonify({})