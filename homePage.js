document.addEventListener("DOMContentLoaded", function () {
    const textElement = document.getElementById("welcome-text");
    const textToAnimate = "Welcome to Plantify: Your Ultimate Plant Enthusiast Hub!";
    let currentIndex = 0;

    function animateText() {
        if (currentIndex < textToAnimate.length) {
            textElement.textContent += textToAnimate[currentIndex];
            currentIndex++;
            setTimeout(animateText, 50); // Adjust the delay (in milliseconds) to control the speed of the animation
        }
    }

    animateText();
});
