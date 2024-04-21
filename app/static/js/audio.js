document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const playBtn = document.getElementById('playBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const visualizer = document.getElementById('visualizer');
    const canvasContext = visualizer.getContext('2d');
    let audioChunks = [];
    let mediaRecorder;
    let audioContext;
    let analyser;
    let dataArray;
    let audio;

    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    playBtn.addEventListener('click', playRecording);
    uploadBtn.addEventListener('click', uploadRecording);

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then((stream) => {
                mediaRecorder = new MediaRecorder(stream);
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                dataArray = new Uint8Array(analyser.frequencyBinCount);

                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audio = new Audio(URL.createObjectURL(audioBlob));
                    playBtn.disabled = false;
                    uploadBtn.disabled = false;
                };

                mediaRecorder.start();

                drawVisualization();
                startBtn.disabled = true;
                stopBtn.disabled = false;
            })
            .catch((error) => {
                console.error('Error accessing microphone:', error);
            });
    }

    function stopRecording() {
        mediaRecorder.stop();
        if (audioContext) audioContext.close();
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }

    function playRecording() {
        if (audio) {
            audio.play();
        }
    }

    function uploadRecording() {
        if (audioChunks.length === 0) {
            console.warn('No audio recorded to upload.');
            return;
        }

        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recorded_audio.wav');

        fetch('/analysis', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            console.log('Audio successfully uploaded!');
        })
        .catch(error => {
            console.error('Error uploading audio:', error);
        });
    }

    function drawVisualization() {
        analyser.getByteTimeDomainData(dataArray);

        canvasContext.clearRect(0, 0, visualizer.width, visualizer.height);
        canvasContext.lineWidth = 2;
        canvasContext.strokeStyle = '#00F';

        canvasContext.beginPath();

        const sliceWidth = visualizer.width / analyser.fftSize;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * visualizer.height / 2;

            if (i === 0) {
                canvasContext.moveTo(x, y);
            } else {
                canvasContext.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasContext.lineTo(visualizer.width, visualizer.height / 2);
        canvasContext.stroke();

        requestAnimationFrame(drawVisualization);
    }
});
