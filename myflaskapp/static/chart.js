
function createChart(probabilities) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(probabilities),
            datasets: [{
                label: 'Probability',
                data: Object.values(probabilities),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const errorDiv = document.getElementById('error');
    const resultDiv = document.getElementById('result');
    const predictionSpan = document.getElementById('prediction');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        // Clear previous results and show loading
        errorDiv.textContent = '';
        resultDiv.style.display = 'none';
        loadingDiv.style.display = 'block';

        const formData = new FormData(form);
        
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingDiv.style.display = 'none';
            if (data.error) {
                errorDiv.textContent = data.error;
                resultDiv.style.display = 'none';
            } else {
                predictionSpan.textContent = data.prediction;
                resultDiv.style.display = 'block';
                createChart(data.probabilities);
            }
        })
        .catch(error => {
            loadingDiv.style.display = 'none';
            console.error('Error:', error);
            errorDiv.textContent = 'An error occurred. Please try again.';
            resultDiv.style.display = 'none';
        });
    });
});

window.onload = function() {
    document.getElementById('upload-form').onsubmit = function(e) {
        var reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('uploadedImage').src = e.target.result;
            document.getElementById('uploadedImage').style.display = 'block';
        }
        reader.readAsDataURL(document.querySelector('input[type=file]').files[0]);
    }
}