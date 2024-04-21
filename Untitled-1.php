<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Recorder</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <script src="https://unpkg.com/wavesurfer.js"></script> <!-- Include WaveSurfer.js -->
    <script src="https://cdn.jsdelivr.net/npm/pitch.js"></script> <!-- Include Pitch.js -->
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        button {
            padding: 10px 20px;
            margin-top: 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:active {
            background-color: #0056b3;
        }
        #waveform {
            width: 100%;
            height: 150px; /* Adjust height as needed */
        }
        #pitch {
            width: 100%;
            height: 150px; /* Adjust height as needed */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Audio Recorder</h1>
        <button id="startBtn"><i class="fas fa-play"></i> Start Recording</button>
        <button id="stopBtn"><i class="fas fa-stop"></i> Stop Recording</button>
        <button id="submitBtn"><i class="fas fa-upload"></i> Submit</button>
        <button id="playBtn"><i class="fas fa-play"></i> Play</button>
        <div id="waveform"></div> <!-- Waveform container -->
        <div id="pitch"></div> <!-- Pitch contour container -->
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const submitBtn = document.getElementById('submitBtn');
        const playBtn = document.getElementById('playBtn');
        const waveformContainer = document.getElementById('waveform');
        const pitchContainer = document.getElementById('pitch');

        let chunks = [];
        let mediaRecorder;
        let wavesurfer;
        let pitch;

        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        submitBtn.addEventListener('click', submitRecording);
        playBtn.addEventListener('click', playRecording);

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = e => chunks.push(e.data);
                    mediaRecorder.start();
                })
                .catch(console.error);
        }

        function stopRecording() {
            mediaRecorder.stop();
        }

        function submitRecording() {
            const blob = new Blob(chunks, { type: 'audio/wav' });

            // Display waveform
            wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: 'blue',
                progressColor: 'purple'
            });
            wavesurfer.loadBlob(blob);

            // Initialize pitch analysis
            pitch = new Pitch();
            pitch.input(wavesurfer.backend.ac);
            pitch.on('pitch', handlePitch);
            console.log(pitch);
        }

        function playRecording() {
            wavesurfer.play();
        }

        function handlePitch(pitch) {
            // Display pitch data here
            console.log(pitch);
        }
    </script>
</body>
</html>

from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io  # Import io module
import base64
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Here you can process the audio data
    # For simplicity, let's just generate some random data for the plots
    audio_data = np.random.rand(1000)  # Replace this with your audio data
    
    # Generate plot
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(audio_data)
    ax[0].set_title('Waveform')
    
    ax[1].specgram(audio_data, Fs=1000)  # Spectrogram
    ax[1].set_title('Spectrogram')
    
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return jsonify({'plot': base64.b64encode(img.getvalue()).decode('utf-8')})

if __name__ == '__main__':
    app.run(debug=True)
