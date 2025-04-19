from flask import Flask, request, send_file
import os
import uuid
import zipfile
import subprocess
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'videos'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
    file.save(video_path)

    extractor_command = [
         'python',  'extractor\CLIFF\demo.py',
        '--input_path', video_path,
        '--input_type', 'video',
        '--ckpt', 'extractor\CLIFF\data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt',
        '--backbone', 'hr48'
        '--batch_size', '1',
        '--save_results'
    ]
    subprocess.run(extractor_command)

    output_zip = f'{UPLOAD_FOLDER}/{video_id}.zip'
    shutil.make_archive(output_zip.replace('.zip', ''), 'zip', UPLOAD_FOLDER)

    return send_file(output_zip, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
