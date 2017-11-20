from flask import Flask, jsonify, request, send_file
from werkzeug import secure_filename
import config as cfg
import boto3, os, time
from boto3.s3.transfer import S3Transfer
from flask_swagger import swagger

AWS_BUCKET = "test-upload-python-s3"
app = Flask("python_s3")
app.debug = True

# upload image api
@app.route("/upload", methods=['POST'])
def upload():
    """
    Upload files
    ---
    tags:
        - Files
    consumes: "multipart/form-data"
    parameters:
        -   name: files[]
            in: formData
            required: true
            paramType: body
            dataType: file
            type: file
    responses:
        200:
            description: Returns album_id after upload
        401:
            description: Unauthorized
        400:
            description: Bad Request
        500:
            description: Server Internal error
    """
    uploaded_files = request.files.getlist("files[]")
    for upload_file in uploaded_files:
        if upload_file and allowed_file(upload_file.filename):
            filename = secure_filename(str(time.time()) + upload_file.filename)

            dir_name = 'uploads/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            file_path = os.path.join(dir_name, filename)

            app.logger.info("Saving file: %s", file_path)
            # save to local 
            upload_file.save(file_path)
            transfer = S3Transfer(boto3.client('s3', cfg.AWS_REGION, aws_access_key_id=cfg.AWS_APP_ID,
                aws_secret_access_key=cfg.AWS_APP_SECRET))

            transfer.upload_file(file_path, AWS_BUCKET, file_path)

    return jsonify({'success': 1})

# down load image
@app.route("/download/<image>", methods=['GET'])
def download(image):
    key = 'uploads/' + secure_filename(image)
    UPLOAD_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], ''))
    # get file from S3
    transfer = S3Transfer(boto3.client('s3', cfg.AWS_REGION, aws_access_key_id=cfg.AWS_APP_ID,
                aws_secret_access_key=cfg.AWS_APP_SECRET))
    # download file from aws
    transfer.download_file(AWS_BUCKET, key, key)

    return send_file(UPLOAD_DIR + "/" + key, mimetype='image/gif')

# Swagger Doccument for API
@app.route('/docs')
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Python S3 API"
    swag['basePath'] = "/"
    return jsonify(swag)

# Cross origin
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin','*')
    response.headers.add('Access-Control-Allow-Headers', "Authorization, Content-Type")
    response.headers.add('Access-Control-Expose-Headers', "Authorization")
    response.headers.add('Access-Control-Allow-Methods', "GET, POST, PUT, DELETE, OPTIONS")
    response.headers.add('Access-Control-Allow-Credentials', "true")
    response.headers.add('Access-Control-Max-Age', 60 * 60 * 24 * 20)
    return response

def allowed_file(filename):
    return '.' in filename and \
       filename.rsplit('.', 1)[1] in cfg.ALLOWED_EXTENSIONS

if (__name__ == "__main__" ):
    app.run('0.0.0.0')
