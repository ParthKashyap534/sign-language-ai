/**
 * NEUROSIGN PRO v2.0 - CORE JAVASCRIPT ENGINE
 * Handles Webcam, MediaPipe Vision, and API Communication
 */

// 1. HTML Elements
const videoElement = document.getElementById('webcam_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

const statusText = document.getElementById('system-status-text');
const indicatorDot = document.querySelector('.indicator-dot');
const fpsDisplay = document.getElementById('fps-display');

const currentWord = document.getElementById('current-word');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-percentage');

const bufferFill = document.getElementById('buffer-fill');
const bufferCount = document.getElementById('buffer-count');

// 2. State Variables
let sequenceBuffer = [];
const SEQUENCE_LENGTH = 30; // Must match our Python Neural Network
let isPredicting = false;
let lastFrameTime = 0;
let frameCount = 0;

// 3. Update UI Buffer Gauge
function updateBufferUI() {
    const percentage = (sequenceBuffer.length / SEQUENCE_LENGTH) * 100;
    bufferFill.style.width = `${percentage}%`;
    bufferCount.innerText = `${sequenceBuffer.length}/${SEQUENCE_LENGTH}`;
}

// 4. Send Data to Python API
async function requestPrediction() {
    if (isPredicting) return;
    isPredicting = true;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: sequenceBuffer })
        });
        
        const data = await response.json();
        
        if (data.action && data.action !== '...') {
            currentWord.innerText = data.action;
            confidenceFill.style.width = `${data.confidence}%`;
            confidenceText.innerText = `${data.confidence}%`;
        } else {
            // Keep UI clean if confidence is low
            confidenceFill.style.width = `0%`;
            confidenceText.innerText = `0%`;
        }
    } catch (error) {
        console.error("API Error:", error);
        currentWord.innerText = "API OFFLINE";
        currentWord.style.color = "#ff003c";
    } finally {
        isPredicting = false;
    }
}

// 5. MediaPipe AI Logic (Fires every time a frame is captured)
function onResults(results) {
    // Calculate FPS
    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        fpsDisplay.innerHTML = `<i class="fa-solid fa-bolt"></i> ${frameCount} FPS`;
        frameCount = 0;
        lastFrameTime = now;
    }

    // Clear Canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // If a hand is detected
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Draw the skeleton
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00f0ff', lineWidth: 2});
            drawLandmarks(canvasCtx, landmarks, {color: '#00ff88', lineWidth: 1, radius: 3});
        }

        // Extract Data
        let frameData = [];
        const targetHand = results.multiHandLandmarks[0]; // Just use the first detected hand
        
        for (let i = 0; i < targetHand.length; i++) {
            frameData.push(targetHand[i].x, targetHand[i].y, targetHand[i].z);
        }

        // Add to buffer
        sequenceBuffer.push(frameData);

        // Keep buffer at exactly 30 frames
        if (sequenceBuffer.length > SEQUENCE_LENGTH) {
            sequenceBuffer.shift(); 
        }

        updateBufferUI();

        // Trigger AI when buffer is full
        if (sequenceBuffer.length === SEQUENCE_LENGTH) {
            requestPrediction();
        }

    } else {
        // If hand is lost, slowly clear the buffer
        if (sequenceBuffer.length > 0) {
            sequenceBuffer.shift();
            updateBufferUI();
        }
    }
    canvasCtx.restore();
}

// 6. Initialize Camera & AI Engine
const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
    maxNumHands: 1, // Only track one hand for now to keep it simple
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});

// Boot up
camera.start().then(() => {
    statusText.innerText = "VISION ENGINE ONLINE";
    indicatorDot.style.background = "#00ff88"; // Turn dot green
    indicatorDot.style.boxShadow = "0 0 10px #00ff88";
});