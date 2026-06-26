class FacialExpressionDetector {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('overlay');
        this.stream = null;
        this.isRunning = false;
        this.animationId = null;
        this.lastDetection = 0;
        this.detectionInterval = 500; // ms between detections
        this.history = [];
        this.maxHistory = 20;

        // DOM elements
        this.emotionDisplay = document.getElementById('emotionDisplay');
        this.confidenceDisplay = document.getElementById('confidenceDisplay');
        this.predictionsList = document.getElementById('predictionsList');
        this.chartContainer = document.querySelector('.chart-bars');
        this.historyList = document.getElementById('historyList');

        // Bind buttons
        document.getElementById('startBtn').addEventListener('click', () => this.start());
        document.getElementById('stopBtn').addEventListener('click', () => this.stop());
        document.getElementById('captureBtn').addEventListener('click', () => this.capture());

        // Emotion emoji mapping
        this.emotionEmojis = {
            'Angry': '😠',
            'Disgust': '🤢',
            'Fear': '😨',
            'Happy': '😊',
            'Sad': '😢',
            'Surprise': '😮',
            'Neutral': '😐'
        };
    }

    async start() {
        try {
            // Request camera
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });

            this.video.srcObject = this.stream;
            await this.video.play();

            // Set canvas size
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;

            // Hide overlay
            this.overlay.classList.add('hidden');

            // Update UI
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            // Start detection loop
            this.isRunning = true;
            this.detectLoop();

        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Unable to access camera. Please ensure you have granted camera permissions.');
            this.overlay.innerHTML = `
                <p style="color: #ff6b6b;">❌ Camera access denied</p>
                <p style="font-size: 0.9rem; margin-top: 10px;">Please allow camera access and refresh</p>
            `;
        }
    }

    stop() {
        this.isRunning = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.video.srcObject = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update UI
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;

        // Show overlay
        this.overlay.classList.remove('hidden');
        this.overlay.innerHTML = `
            <div class="loading-spinner"></div>
            <p>Camera stopped</p>
        `;
    }

    async capture() {
        if (!this.isRunning) {
            alert('Please start the camera first!');
            return;
        }

        // Draw current frame on canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Get image data
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
        
        // Create download link
        const link = document.createElement('a');
        link.download = `expression_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.jpg`;
        link.href = imageData;
        link.click();
    }

    detectLoop() {
        if (!this.isRunning) return;

        const now = Date.now();
        if (now - this.lastDetection >= this.detectionInterval) {
            this.detectExpression();
            this.lastDetection = now;
        }

        // Draw video frame with face box
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        this.drawFaceBox();

        this.animationId = requestAnimationFrame(() => this.detectLoop());
    }

    drawFaceBox() {
        // Draw a simple face detection box in the center
        // In production, you'd use a proper face detection model
        const width = this.canvas.width;
        const height = this.canvas.height;
        const boxWidth = width * 0.6;
        const boxHeight = height * 0.6;
        const x = (width - boxWidth) / 2;
        const y = (height - boxHeight) / 2;

        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, boxWidth, boxHeight);
        
        // Add corner markers
        const cornerLength = 20;
        const corners = [
            [x, y, 1, 1],
            [x + boxWidth, y, -1, 1],
            [x, y + boxHeight, 1, -1],
            [x + boxWidth, y + boxHeight, -1, -1]
        ];
        
        corners.forEach(([cx, cy, dx, dy]) => {
            this.ctx.beginPath();
            this.ctx.moveTo(cx + dx * cornerLength, cy);
            this.ctx.lineTo(cx, cy);
            this.ctx.lineTo(cx, cy + dy * cornerLength);
            this.ctx.stroke();
        });
    }

    async detectExpression() {
        try {
            // Capture frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.5);

            // Send to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (result.error) {
                console.error('Prediction error:', result.error);
                return;
            }

            this.updateUI(result);

        } catch (error) {
            console.error('Detection error:', error);
        }
    }

    updateUI(result) {
        const { emotion, confidence, top_predictions, all_probabilities } = result;

        // Update emotion display
        const emoji = this.emotionEmojis[emotion] || '🤔';
        this.emotionDisplay.innerHTML = `
            <span class="emotion-icon">${emoji}</span>
            <span class="emotion-text">${emotion}</span>
        `;
        this.confidenceDisplay.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

        // Update predictions list
        this.predictionsList.innerHTML = top_predictions.map((p, index) => `
            <div class="prediction-item ${index === 0 ? 'highlight' : ''}">
                <span class="prediction-label">${this.emotionEmojis[p.emotion] || ''} ${p.emotion}</span>
                <span class="prediction-confidence">${(p.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');

        // Update probability chart
        this.updateChart(all_probabilities);

        // Update history
        this.addToHistory(emotion, confidence);
    }

    updateChart(probabilities) {
        const maxProb = Math.max(...Object.values(probabilities));
        
        this.chartContainer.innerHTML = Object.entries(probabilities).map(([emotion, prob]) => {
            const percentage = (prob / maxProb) * 100;
            const height = Math.max(percentage * 0.8, 5); // Min height for visibility
            const emoji = this.emotionEmojis[emotion] || '';
            
            return `
                <div class="bar-wrapper">
                    <div class="bar" style="height: ${height}%; background: linear-gradient(180deg, #667eea ${height}%, #764ba2 100%);">
                        <span style="position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 0.7rem; color: #333; font-weight: 600;">
                            ${(prob * 100).toFixed(0)}%
                        </span>
                    </div>
                    <div class="bar-label">${emoji}</div>
                </div>
            `;
        }).join('');
    }

    addToHistory(emotion, confidence) {
        const entry = `${new Date().toLocaleTimeString()} - ${emotion} (${(confidence * 100).toFixed(0)}%)`;
        this.history.unshift(entry);
        if (this.history.length > this.maxHistory) {
            this.history.pop();
        }

        this.historyList.innerHTML = this.history.map(item => 
            `<li>${item}</li>`
        ).join('') || '<li>No detections yet</li>';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const detector = new FacialExpressionDetector();
    console.log('Facial Expression Detector initialized!');
});
