function predictPrice() {
    fetch('/predict', {
        method: 'POST',
        body: new FormData(document.getElementById('predictForm'))
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('prediction-result');
        if (data.error) {
            resultDiv.className = 'result-message error-message';
            resultDiv.textContent = 'Error: ' + data.error;
        } else {
            resultDiv.className = 'result-message success-message';
            resultDiv.textContent = 'Predicted Car Price: ' + data.prediction;
        }
    });
}

function trainModel() {
    fetch('/train', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        const statusDiv = document.getElementById('training-status');
        if (data.error) {
            statusDiv.className = 'status-message error-message';
            statusDiv.textContent = 'Error: ' + data.error;
        } else {
            statusDiv.className = 'status-message success-message';
            statusDiv.textContent = data.message;
            location.reload(); // Simple way to reflect model availability
        }
    });
}