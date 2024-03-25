// Define your disease labels based on your model's output
const diseaseLabels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    
    
    // Add more disease names here based on your model's output
];

// async function checkClassOrder() {
//     // Load the TensorFlow.js model JSON file
//     const model = await tf.loadLayersModel('tfjs_model/model.json');

//     // Display the model summary
//     model.summary();
// }

// // Call the function to check the class order
// checkClassOrder();


async function identifyDisease() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image.');
        return;
    }

    // Load the TensorFlow.js model
    const model = await tf.loadLayersModel('tfjs_model/model.json');

    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = async function () {
        const tensor = tf.browser.fromPixels(image);
        const resizedTensor = tf.image.resizeBilinear(tensor, [224, 224]).toFloat();
        const normalizedTensor = resizedTensor.div(255.0); // Normalize pixel values
        const expandedTensor = normalizedTensor.expandDims();

        // Make predictions
        const predictions = await model.predict(expandedTensor);

        // Convert predictions to a JavaScript array
        const predictedProbabilities = Array.from(predictions.dataSync());

        // Find the index of the class with the highest probability
        const diseaseIndex = predictedProbabilities.indexOf(Math.max(...predictedProbabilities));

        // Get the predicted disease name and confidence score
        const diseaseName = diseaseLabels[diseaseIndex];
        const confidenceScore = predictedProbabilities[diseaseIndex];


      

        displayResult(diseaseName, confidenceScore);
    };
}


function displayResult(diseaseName, confidenceScore) {
    const resultDiv = document.getElementById('results');
    resultDiv.innerHTML = ''; // Clear previous results
  
    const predictedDisease = document.createElement('p');
    predictedDisease.innerHTML = `<strong>Predicted Disease:</strong> ${diseaseName}`;
  
    const confidenceScoreText = document.createElement('p');
    confidenceScoreText.innerHTML = `<strong>Confidence Score:</strong> ${(confidenceScore * 100).toFixed(2)}%`;
  
    // Append elements to the resultDiv
    resultDiv.appendChild(predictedDisease);
    resultDiv.appendChild(confidenceScoreText);
  }
