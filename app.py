from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file to a directory (e.g., 'uploads')
            file.save('uploads/' + file.filename)
            return redirect(url_for('success'))
    return render_template('form.php')

@app.route('/success')
def success():
    return "File uploaded successfully!"

if __name__ == '__main__':
    app.run(debug=True)
