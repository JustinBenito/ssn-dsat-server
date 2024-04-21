document.addEventListener('DOMContentLoaded', () => {
    let audioContext;
    let recorder;
    let wavesurfer;

    const startRecordingButton = document.getElementById('startRecording');
    const stopRecordingButton = document.getElementById('stopRecording');
    const playbackButton = document.getElementById('playback');
    const waveformContainer = document.getElementById('waveform');

    startRecordingButton.addEventListener('click', startRecording);
    stopRecordingButton.addEventListener('click', stopRecording);
    playbackButton.addEventListener('click', playback);

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                recorder = new Recorder(audioContext.createMediaStreamSource(stream));

                // Initialize WaveSurfer
                wavesurfer = WaveSurfer.create({
                    container: '#waveform',
                    waveColor: 'violet',
                    progressColor: 'purple',
                    cursorColor: 'white',
                    height: 100,
                });

                // Connect the audio source to WaveSurfer
                recorder.source.connect(wavesurfer.backend.analyser);

                // Start the WaveSurfer visualization
                wavesurfer.clearRegions();
                wavesurfer.clearMarks();
                wavesurfer.empty();
                wavesurfer.load(stream);
                wavesurfer.play();

                recorder.record();

                // Update the waveform in real-time
                setInterval(() => {
                    wavesurfer.empty();
                    wavesurfer.load(stream);
                }, 100);
            })
            .catch(error => console.error('Error accessing microphone:', error));
    }

    function stopRecording() {
        if (recorder) {
            recorder.stop();
            recorder.exportWAV(createDownloadLink);
        }
    }

    function createDownloadLink(blob) {
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        document.body.appendChild(audio);

        // Stop the WaveSurfer visualization
        wavesurfer.stop();
    }

    function playback() {
        if (recorder) {
            recorder.play();
        }
    }
});