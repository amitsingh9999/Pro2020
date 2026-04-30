document.addEventListener('DOMContentLoaded', () => {

    // --- Mobile Navigation Toggle ---
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    const navLinks = document.querySelectorAll('.nav-links li');

    burger.addEventListener('click', () => {
        nav.classList.toggle('nav-active');
        navLinks.forEach((link, index) => {
            if (link.style.animation) {
                link.style.animation = '';
            } else {
                link.style.animation =
                    `navLinkFade 0.5s ease forwards ${index / 7 + 0.3}s`;
            }
        });
        burger.classList.toggle('toggle');
    });

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            nav.classList.remove('nav-active');
            burger.classList.remove('toggle');
            navLinks.forEach(item => {
                item.style.animation = '';
            });
        });
    });

    // --- Dark Theme Toggle ---
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
        themeToggle.textContent = '☀️';
    } else {
        themeToggle.textContent = '🌙';
    }

    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-theme');
        if (body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.textContent = '☀️';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.textContent = '🌙';
        }
    });

    // --- Image Upload ---
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const previewImg = document.getElementById('preview-img');
    const fileNameSpan = document.getElementById('file-name');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const diagnoseBtn = document.getElementById('diagnose-btn');
    const diagnosisResult = document.getElementById('diagnosis-result');

    let uploadedFile = null;

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file) return;

        if (!file.type.match('image.*')) {
            alert("Only image files allowed!");
            return;
        }

        uploadedFile = file;
        fileNameSpan.textContent = file.name;
        diagnoseBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewImg.style.display = "block";
        };
        reader.readAsDataURL(file);

        removeImgBtn.style.display = "inline-block";
        diagnosisResult.innerHTML = "";
    }

    // ⭐⭐⭐ --- REAL AI PREDICTION CALL (FASTAPI) --- ⭐⭐⭐
diagnoseBtn.addEventListener('click', async () => {
    if (!uploadedFile) return;

    diagnosisResult.innerHTML = "🧠 Analyzing image... please wait...";
    diagnosisResult.className = "diagnosis-result loading";
    diagnoseBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        let resultClass = "positive";
        let extraMsg = "⚠️ Possible disease detected";

        if (result.prediction === "Normal") {
            resultClass = "negative";
            extraMsg = "✅ No disease detected";
        }

        diagnosisResult.className = "diagnosis-result " + resultClass;

        diagnosisResult.innerHTML = `
            <h4>🩺 Result: <strong>${result.prediction}</strong></h4>
            <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
            <p>${extraMsg}</p>
            <p><em>This is AI-based prediction. Consult a doctor.</em></p>
        `;

    } catch (error) {
        diagnosisResult.className = "diagnosis-result";
        diagnosisResult.innerHTML =
            "❌ Error connecting to backend. Make sure Flask is running.";
    }

    diagnoseBtn.disabled = false;
});

    // Remove Image
    removeImgBtn.addEventListener('click', () => {
        uploadedFile = null;
        fileInput.value = "";
        previewImg.src = "";
        previewImg.style.display = "none";
        fileNameSpan.textContent = "No file chosen";
        removeImgBtn.style.display = "none";
        diagnoseBtn.disabled = true;
        diagnosisResult.innerHTML = "";
    });

    dropArea.addEventListener("click", () => fileInput.click());
    ['dragenter','dragover','dragleave','drop'].forEach(ev =>
        dropArea.addEventListener(ev, preventDefaults)
    );

    dropArea.addEventListener("drop", e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener("change", e => handleFiles(e.target.files));

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href'))
                .scrollIntoView({ behavior: "smooth" });
        });
    });

});
