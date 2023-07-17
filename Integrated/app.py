# parent_app.py
from flask import Flask, render_template, request, redirect, url_for
from demo1.app import app1_bp
from demo2.app import app2_bp
from demo3.app import app3_bp

app = Flask(__name__, static_url_path='/static')

# Register the blueprints
app.register_blueprint(app1_bp)
app.register_blueprint(app2_bp)
app.register_blueprint(app3_bp)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    selected_option = request.form.get('option')
    if selected_option == 'option1':
        return redirect(url_for('app1.app1_route'))
    elif selected_option == 'option2':
        return redirect(url_for('app2.app2_route'))
    elif selected_option == 'option3':
        return redirect(url_for('app3.app3_route'))
    else:
        return "Invalid option!"

# Run the parent app
if __name__ == "__main__":
    app.run()
