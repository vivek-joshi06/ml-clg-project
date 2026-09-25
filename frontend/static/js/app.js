/**
 * CardioScan AI — 3-Section Showcase JavaScript Logic
 * Handles smooth scrolling, dynamic vitals calculation, SVG gauge animations,
 * active section observer, preset profile loaders, and inference REST API.
 */

// Application State
const appState = {
    gender: 1,
    cholesterol: 1,
    gluc: 1,
    smoke: 0,
    alco: 0,
    active: 1,
    featureImportances: []
};

// Preset Patient Profiles
const PRESETS = {
    low: {
        age: 32,
        gender: 1,
        height: 166.0,
        weight: 58.0,
        ap_hi: 112,
        ap_lo: 72,
        cholesterol: 1,
        gluc: 1,
        smoke: 0,
        alco: 0,
        active: 1
    },
    moderate: {
        age: 52,
        gender: 2,
        height: 174.0,
        weight: 84.0,
        ap_hi: 138,
        ap_lo: 88,
        cholesterol: 2,
        gluc: 1,
        smoke: 1,
        alco: 0,
        active: 1
    },
    high: {
        age: 62,
        gender: 2,
        height: 170.0,
        weight: 94.0,
        ap_hi: 165,
        ap_lo: 104,
        cholesterol: 3,
        gluc: 2,
        smoke: 1,
        alco: 1,
        active: 0
    }
};

document.addEventListener("DOMContentLoaded", () => {
    initScrollObserver();
    fetchModelInfo();
    updateLiveCalculations();
    executePrediction(); // Initialize with default prediction
});

/**
 * Setup IntersectionObserver for active section highlight in navbar and side dots
 */
function initScrollObserver() {
    const sections = document.querySelectorAll(".screen-section");
    const navItems = document.querySelectorAll(".nav-item");
    const dotLinks = document.querySelectorAll(".dot-link");

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute("id");

                    navItems.forEach((link) => {
                        if (link.getAttribute("data-section") === id) {
                            link.classList.add("active");
                        } else {
                            link.classList.remove("active");
                        }
                    });

                    dotLinks.forEach((dot) => {
                        if (dot.getAttribute("data-section") === id) {
                            dot.classList.add("active");
                        } else {
                            dot.classList.remove("active");
                        }
                    });
                }
            });
        },
        { threshold: 0.55 }
    );

    sections.forEach((sec) => observer.observe(sec));
}

/**
 * Fetch model information & feature importances from Flask API
 */
async function fetchModelInfo() {
    try {
        const res = await fetch("/api/info");
        const json = await res.json();
        if (json.status === "success") {
            appState.featureImportances = json.data.feature_importances;
            renderFeatureBars(json.data.feature_importances);
        }
    } catch (err) {
        console.error("Failed to load /api/info:", err);
    }
}

/**
 * Render feature importance horizontal bars in Section 3
 */
function renderFeatureBars(features) {
    const container = document.getElementById("featureBarsContainer");
    if (!container) return;

    container.innerHTML = "";
    const maxVal = Math.max(...features.map((f) => f.importance));

    features.forEach((f) => {
        const pctWidth = (f.importance / maxVal) * 100;
        const row = document.createElement("div");
        row.className = "feature-bar-row";
        row.innerHTML = `
            <span class="feature-name" title="${f.feature}">${f.feature}</span>
            <div class="feature-track">
                <div class="feature-fill" style="width: ${pctWidth}%;"></div>
            </div>
            <span class="feature-pct">${f.importance.toFixed(1)}%</span>
        `;
        container.appendChild(row);
    });
}

/**
 * Handle Toggle Pill Group Selections
 */
function selectPill(groupId, value) {
    const container = document.getElementById(groupId);
    if (!container) return;

    const buttons = container.querySelectorAll(".t-pill");
    buttons.forEach((btn) => {
        if (parseInt(btn.getAttribute("data-val")) === value) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    if (groupId === "genderGroup") appState.gender = value;
    else if (groupId === "cholGroup") appState.cholesterol = value;
    else if (groupId === "glucGroup") appState.gluc = value;
    else if (groupId === "smokeGroup") appState.smoke = value;
    else if (groupId === "alcoGroup") appState.alco = value;
    else if (groupId === "activeGroup") appState.active = value;
}

/**
 * Load Quick Clinical Presets
 */
function loadPreset(key) {
    const p = PRESETS[key];
    if (!p) return;

    document.getElementById("age").value = p.age;
    document.getElementById("height").value = p.height;
    document.getElementById("weight").value = p.weight;
    document.getElementById("ap_hi").value = p.ap_hi;
    document.getElementById("ap_lo").value = p.ap_lo;

    selectPill("genderGroup", p.gender);
    selectPill("cholGroup", p.cholesterol);
    selectPill("glucGroup", p.gluc);
    selectPill("smokeGroup", p.smoke);
    selectPill("alcoGroup", p.alco);
    selectPill("activeGroup", p.active);

    updateLiveCalculations();
    executePrediction();
}

/**
 * Real-time dynamic calculation of BMI and BP Staging
 */
function updateLiveCalculations() {
    const height = parseFloat(document.getElementById("height").value) || 170;
    const weight = parseFloat(document.getElementById("weight").value) || 70;
    const ap_hi = parseFloat(document.getElementById("ap_hi").value) || 120;
    const ap_lo = parseFloat(document.getElementById("ap_lo").value) || 80;

    // Calculate BMI
    const heightM = height / 100.0;
    const bmi = (weight / (heightM * heightM)).toFixed(1);
    const bmiDisplay = document.getElementById("bmiDisplay");
    const bmiBadge = document.getElementById("bmiBadge");

    if (bmiDisplay && bmiBadge) {
        bmiDisplay.textContent = `${bmi} kg/m²`;
        if (bmi < 18.5) {
            bmiBadge.textContent = "Underweight";
            bmiBadge.className = "rt-chip";
            bmiBadge.style.color = "#38bdf8";
            bmiBadge.style.background = "rgba(56, 189, 248, 0.15)";
        } else if (bmi < 25.0) {
            bmiBadge.textContent = "Normal";
            bmiBadge.className = "rt-chip normal";
            bmiBadge.style = "";
        } else if (bmi < 30.0) {
            bmiBadge.textContent = "Overweight";
            bmiBadge.className = "rt-chip warning";
            bmiBadge.style = "";
        } else {
            bmiBadge.textContent = "Obese";
            bmiBadge.className = "rt-chip danger";
            bmiBadge.style = "";
        }
    }

    // Calculate BP Stage & Pulse Pressure
    const pulsePressure = Math.round(ap_hi - ap_lo);
    const bpDisplay = document.getElementById("bpDisplay");
    const bpBadge = document.getElementById("bpBadge");

    if (bpDisplay && bpBadge) {
        bpDisplay.textContent = `Pulse: ${pulsePressure} mmHg`;

        if (ap_hi < 120 && ap_lo < 80) {
            bpBadge.textContent = "Normal BP";
            bpBadge.className = "rt-chip normal";
        } else if (ap_hi < 130 && ap_lo < 80) {
            bpBadge.textContent = "Elevated BP";
            bpBadge.className = "rt-chip warning";
        } else if ((ap_hi >= 130 && ap_hi < 140) || (ap_lo >= 80 && ap_lo < 90)) {
            bpBadge.textContent = "Stage 1 HTN";
            bpBadge.className = "rt-chip warning";
        } else {
            bpBadge.textContent = "Stage 2 HTN";
            bpBadge.className = "rt-chip danger";
        }
    }
}

/**
 * Handle Form Submission
 */
function handlePrediction(event) {
    if (event) event.preventDefault();
    executePrediction();
}

/**
 * Run Inference Request to Flask Backend
 */
async function executePrediction() {
    const age = parseFloat(document.getElementById("age").value);
    const height = parseFloat(document.getElementById("height").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const ap_hi = parseFloat(document.getElementById("ap_hi").value);
    const ap_lo = parseFloat(document.getElementById("ap_lo").value);

    const errBox = document.getElementById("formErrorMessage");
    if (ap_lo >= ap_hi) {
        errBox.style.display = "block";
        errBox.textContent = "⚠️ Error: Diastolic pressure (ap_lo) cannot be equal to or greater than Systolic pressure (ap_hi).";
        return;
    }
    errBox.style.display = "none";

    const btn = document.getElementById("submitBtn");
    const spinner = document.getElementById("btnSpinner");
    if (spinner) spinner.style.display = "inline-block";
    if (btn) btn.disabled = true;

    const payload = {
        age: age,
        gender: appState.gender,
        height: height,
        weight: weight,
        ap_hi: ap_hi,
        ap_lo: ap_lo,
        cholesterol: appState.cholesterol,
        gluc: appState.gluc,
        smoke: appState.smoke,
        alco: appState.alco,
        active: appState.active
    };

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const res = await response.json();
        if (res.status === "success") {
            updateResultCard(res);
        } else {
            errBox.style.display = "block";
            errBox.textContent = `⚠️ Server error: ${res.message}`;
        }
    } catch (e) {
        errBox.style.display = "block";
        errBox.textContent = "⚠️ Connection error: Failed to reach inference server.";
        console.error(e);
    } finally {
        if (spinner) spinner.style.display = "none";
        if (btn) btn.disabled = false;
    }
}

/**
 * Update UI Result Card & Animate Speedometer Gauge
 */
function updateResultCard(res) {
    const card = document.getElementById("resultCard");
    const badge = document.getElementById("riskBadge");
    const confLabel = document.getElementById("confidenceLabel");
    const title = document.getElementById("riskTierTitle");
    const desc = document.getElementById("riskDescription");
    const probNum = document.getElementById("probabilityNumber");
    const advisory = document.getElementById("advisoryText");

    card.className = `glass-card result-card tier-${res.risk_badge}`;
    badge.className = `risk-pill ${res.risk_badge}`;
    badge.textContent = res.risk_tier;

    confLabel.textContent = `Confidence: ${res.confidence}%`;
    title.textContent = `${res.risk_tier} Detected`;
    desc.textContent = res.recommendation;
    probNum.textContent = `${res.probability_percent}%`;
    advisory.textContent = res.recommendation;

    // Animate Speedometer Needle & Arc
    animateGauge(res.probability);

    // Populate Contributing Drivers
    const driversList = document.getElementById("driversList");
    driversList.innerHTML = "";
    if (res.contributing_drivers && res.contributing_drivers.length > 0) {
        res.contributing_drivers.forEach((d) => {
            const span = document.createElement("span");
            span.className = "driver-tag";
            span.innerHTML = `⚠️ ${d}`;
            driversList.appendChild(span);
        });
    } else {
        const span = document.createElement("span");
        span.className = "driver-tag optimal";
        span.textContent = "✅ Optimal Biomarkers: No significant risk elevations identified.";
        driversList.appendChild(span);
    }
}

/**
 * Animate SVG Radial Gauge
 */
function animateGauge(probability) {
    const p = Math.min(Math.max(probability, 0), 1);

    // Total arc circumference for radius 95 = 298.4
    const totalArc = 298;
    const offset = totalArc - p * totalArc;

    const activeArc = document.getElementById("gaugeProgressArc");
    if (activeArc) {
        activeArc.style.strokeDashoffset = offset;
        if (p < 0.35) activeArc.style.stroke = "#10b981";
        else if (p < 0.65) activeArc.style.stroke = "#f59e0b";
        else activeArc.style.stroke = "#ef4444";
    }
}
