


let mediaRecorder;
let chunks = [];
let wavesurfer;
const startBtn = document.getElementById('startBtn');
const uploadBtn = document.getElementById('uploadBtn');
const submitBtn = document.getElementById('submitBtn');
const playBtn = document.getElementById('plyBtn');
const waveformContainer = document.getElementById('waveform');
const pitchContainer = document.getElementById('pitch');
const energyContainer = document.getElementById('energy');



startBtn.addEventListener('click', startRecording);
uploadBtn.addEventListener('click', uploadRecording);
submitBtn.addEventListener('click', submitRecording);

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.start();
            startBtn.disabled = true;
            submitBtn.disabled = false;
        })
        .catch(error => console.error('Error starting recording:', error));
}

function stopRecording() {
    mediaRecorder.stop();
}

function uploadRecording() {
    stopRecording();
    const audioData = chunks.map(chunk => new Uint8Array(chunk));
    const blob = new Blob(audioData, { type: 'audio/wav' });

    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');

    fetch('/analysis1', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            console.log('Audio uploaded successfully');
        } else {
            console.error('Failed to upload audio');
        }
    })
    .catch(error => console.error('Error uploading audio:', error));
}

function submitRecording() {
    const blob = new Blob(chunks, { type: 'audio/wav' });

    // Display waveform
    wavesurfer = WaveSurfer.create({
        container: waveformContainer,
        waveColor: 'blue',
        progressColor: 'purple'
    });

    wavesurfer.loadBlob(blob);
    audioData = blob;
    calculatePitchAndEnergy();
    // Extract audio data
    //wavesurfer.backend.buffer = null; // Reset backend buffer to force reload
    //wavesurfer.loadBlob(blob);
    

 
}

function calculatePitchAndEnergy() {
    try {
        // Calculate pitch contour
        const hopLength = 512;
        const pitch = librosa.yin(audioData, {
            fmin: librosa.note_to_hz('C1'),
            fmax: librosa.note_to_hz('C7'),
            sr: wavesurfer.backend.ac.sampleRate,
            hop_length: hopLength
        });
        pitchContainer.innerHTML = ''; // Clear previous pitch contour
        pitchContainer.appendChild(librosa.display.waveshow(pitch, sr = wavesurfer.backend.ac.sampleRate));

        // Calculate energy plot
        const energy = librosa.feature.rms(y = audioData, frame_length = hopLength, hop_length = hopLength);
        energyContainer.innerHTML = ''; // Clear previous energy plot
        energyContainer.appendChild(librosa.display.waveshow(energy, sr = wavesurfer.backend.ac.sampleRate));
    } catch (error) {
        console.error('Error calculating pitch and energy:', error);
    }
}


function plotPitch(pitches) {
    const pitchContainer = document.getElementById('pitch');
    pitchContainer.innerHTML = ''; // Clear previous content

    const plot = document.createElement('div');
    plot.style.width = '100%';
    plot.style.height = '150px';
    plot.style.border = '1px solid black';

    const pitchHeight = 150 / pitches.length;
    pitches.forEach((pitch, index) => {
        const line = document.createElement('div');
        const lineHeight = Math.min(pitch * 2, 150);
        line.style.width = `${100 / pitches.length}%`;
        line.style.height = `${lineHeight}px`;
        line.style.backgroundColor = 'blue';
        line.style.float = 'left';
        plot.appendChild(line);
    });

    pitchContainer.appendChild(plot);
}

// Event listener for the play button
playBtn.addEventListener('click', function() {
    wavesurfer.play(); // Play the waveform
});
 // Function to toggle content
 function toggleContent(buttonId, contentId) {
    var button = document.getElementById(buttonId);
    var content = document.getElementById(contentId);

    // Toggle content display
    if (content.style.display === "none") {
        content.style.display = "block";
    } else {
        content.style.display = "none";
    }
}

// Assign onclick event to each toggle button
document.getElementById("togglePhonation").onclick = function() {
    toggleContent("togglePhonation", "phonationTable");
};
document.getElementById("toggleHNR").onclick = function() {
    toggleContent("toggleHNR", "HNRTable");
};
document.getElementById("toggleF0").onclick = function() {
    toggleContent("toggleF0", "F0Table");
};
document.getElementById("toggleSpectral").onclick = function() {
    toggleContent("toggleSpectral", "spectralTable");
};
document.getElementById("toggleMel").onclick = function() {
    toggleContent("toggleMel", "melTable");
};
function reloadPage() {
    // Get the current URL
    var currentUrl = window.location.href;
    
    // Append query parameter with the div ID to the URL
    var divId = "out"; // Replace with your target div ID
    var urlWithQueryParam = currentUrl + "?scrollTo=" + divId;
    
    // Redirect to the updated URL
    window.location.href = urlWithQueryParam;
  }
  // Check for the query parameter on page load
window.onload = function() {
    var urlParams = new URLSearchParams(window.location.search);
    var scrollToDivId = urlParams.get('out');
    
    if (scrollToDivId) {
        
      var element = document.getElementById(scrollToDivId);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
      }
    }
  };
