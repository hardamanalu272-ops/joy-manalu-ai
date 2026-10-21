/**
 * JOY MANALU AI STUDIO - CLIENT CONTROLLER
 * Real-time Facial Recognition, Cross-Device (Web, Laptop, HP) support,
 * Mobile Camera Flipping, and Vector Telemetry.
 */

// ──────────────── Configuration & State ────────────────
// Otomatis gunakan origin halaman saat ini (selalu sinkron dengan Cloudflare maupun localhost)
let API_BASE = (window.location.protocol.startsWith("http")) ? window.location.origin : "http://localhost:5000";

// Jika ada konfigurasi khusus di config.js
if (window.JOY_AI_CONFIG && window.JOY_AI_CONFIG.BACKEND_API_URL && window.JOY_AI_CONFIG.BACKEND_API_URL.trim() !== "") {
    API_BASE = window.JOY_AI_CONFIG.BACKEND_API_URL.trim().replace(/\/+$/, "");
}

let activeMode = "webcam"; // 'webcam' | 'upload'
let webcamStream = null;
let isDetecting = false;
let currentFacingMode = "user"; // 'user' (selfie/front) | 'environment' (back/rear)
let personsDatabase = [];
let detectionIntervalTimer = null;
let isRequestInFlight = false;

// DOM Elements
const btnWebcam = document.getElementById("btnWebcam");
const btnUpload = document.getElementById("btnUpload");
const webcamWrapper = document.getElementById("webcamWrapper");
const uploadWrapper = document.getElementById("uploadWrapper");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const webcamVideo = document.getElementById("webcamVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const uploadOverlayCanvas = document.getElementById("uploadOverlayCanvas");
const scanLine = document.getElementById("scanLine");
const btnStartCam = document.getElementById("btnStartCam");
const btnStopCam = document.getElementById("btnStopCam");
const btnFlipCam = document.getElementById("btnFlipCam");
const camFacingLabel = document.getElementById("camFacingLabel");
const resultsList = document.getElementById("resultsList");
const faceCountBadge = document.getElementById("faceCount");
const processingIndicator = document.getElementById("processingIndicator");
const perfTime = document.getElementById("perfTime");
const navStatus = document.getElementById("navStatus");
const statPersons = document.getElementById("statPersons");
const personsGrid = document.getElementById("personsGrid");
const httpsWarningBanner = document.getElementById("httpsWarningBanner");
const mobileModal = document.getElementById("mobileModal");
const mobileUrlInput = document.getElementById("mobileUrlInput");

// ──────────────── Initialization ────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkServerHealth();
    loadPersonsVault();
    checkSecureContext();
    setupDropzone();
});

// Check if running on insecure mobile HTTP
function checkSecureContext() {
    const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    if (!window.isSecureContext && !isLocal) {
        if (httpsWarningBanner) httpsWarningBanner.classList.remove("hidden");
    }
}

// ──────────────── Backend Status & Health ────────────────
async function checkServerHealth() {
    const dot = navStatus.querySelector(".status-dot");
    const text = navStatus.querySelector(".status-text");

    try {
        const res = await fetch(`${API_BASE}/api/status`, { method: "GET" });
        if (res.ok) {
            const data = await res.json();
            dot.className = "status-dot online";
            text.textContent = "AI Online";
            if (statPersons) statPersons.textContent = data.registered_persons || "22";
            if (mobileUrlInput && data.mobile_url) {
                mobileUrlInput.value = data.mobile_url;
            }
        } else {
            throw new Error("Server not responding");
        }
    } catch (err) {
        dot.className = "status-dot offline";
        text.textContent = "Offline (Klik)";
    }
}

function configureServerUrl() {
    const current = localStorage.getItem("JOY_AI_API_URL") || API_BASE;
    const input = prompt("Konfigurasi URL Backend Python AI Studio:", current);
    if (input && input.trim() !== "") {
        API_BASE = input.trim().replace(/\/+$/, "");
        localStorage.setItem("JOY_AI_API_URL", API_BASE);
        checkServerHealth();
        loadPersonsVault();
    }
}

// ──────────────── Mode Switching ────────────────
function switchMode(mode) {
    activeMode = mode;
    if (mode === "webcam") {
        btnWebcam.classList.add("active");
        btnUpload.classList.remove("active");
        webcamWrapper.classList.remove("hidden");
        uploadWrapper.classList.add("hidden");
        document.getElementById("panelTitle").textContent = "Kamera AI";
    } else {
        btnUpload.classList.add("active");
        btnWebcam.classList.remove("active");
        webcamWrapper.classList.add("hidden");
        uploadWrapper.classList.remove("hidden");
        document.getElementById("panelTitle").textContent = "Scan Foto Wajah";
        stopWebcam();
    }
    clearResults();
}

// ──────────────── Live Webcam Handling ────────────────
async function startWebcam() {
    try {
        if (webcamStream) {
            stopWebcam();
        }

        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = webcamStream;
        await webcamVideo.play();

        // UI state changes
        videoPlaceholder.classList.add("hidden");
        btnStartCam.classList.add("hidden");
        btnStopCam.classList.remove("hidden");
        scanLine.classList.add("active");

        // Sync overlay canvas dimension
        overlayCanvas.width = webcamVideo.videoWidth || 640;
        overlayCanvas.height = webcamVideo.videoHeight || 480;

        // Update camera facing label
        if (camFacingLabel) {
            camFacingLabel.textContent = currentFacingMode === "user" ? "Depan (Selfie)" : "Belakang (Rear)";
        }

        isDetecting = true;
        startRecognitionLoop();

    } catch (err) {
        console.error("Camera access failed:", err);
        let msg = "Gagal mengakses kamera: " + err.message;
        if (!window.isSecureContext && window.location.hostname !== "localhost") {
            msg = "Kamera diblokir oleh browser karena koneksi bukan HTTPS.\nSilakan gunakan tab 'Scan Foto' atau aktifkan HTTPS / Cloudflare Tunnel.";
        }
        alert(msg);
    }
}

function stopWebcam() {
    isDetecting = false;
    if (detectionIntervalTimer) {
        clearTimeout(detectionIntervalTimer);
        detectionIntervalTimer = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    videoPlaceholder.classList.remove("hidden");
    btnStartCam.classList.remove("hidden");
    btnStopCam.classList.add("hidden");
    scanLine.classList.remove("active");

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Mobile Camera Flip (Depan / Belakang)
async function toggleCameraFacing() {
    currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
    if (camFacingLabel) {
        camFacingLabel.textContent = (currentFacingMode === "user") ? "Depan (Selfie)" : "Belakang (Rear)";
    }
    // Mirror front camera, unmirror rear camera
    webcamVideo.style.transform = (currentFacingMode === "user") ? "scaleX(-1)" : "scaleX(1)";

    if (webcamStream) {
        await startWebcam();
    }
}

// ──────────────── Continuous Recognition Loop ────────────────
function startRecognitionLoop() {
    if (!isDetecting) return;

    // Run frame capture every ~350ms to maintain fluid performance without server congestion
    detectionIntervalTimer = setTimeout(async () => {
        if (!isDetecting) return;
        if (!isRequestInFlight && webcamVideo.readyState === 4) {
            await captureAndRecognizeFrame();
        }
        startRecognitionLoop();
    }, 350);
}

async function captureAndRecognizeFrame() {
    isRequestInFlight = true;

    try {
        // Offscreen canvas for frame capture
        const tempCanvas = document.createElement("canvas");
        const w = webcamVideo.videoWidth;
        const h = webcamVideo.videoHeight;
        
        // Scale down large camera frames to max 720 width for blazing-fast inference
        const scale = Math.min(1.0, 720 / w);
        tempCanvas.width = Math.round(w * scale);
        tempCanvas.height = Math.round(h * scale);

        const ctx = tempCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, tempCanvas.width, tempCanvas.height);

        const b64Image = tempCanvas.toDataURL("image/jpeg", 0.75);

        const res = await fetch(`${API_BASE}/api/recognize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: b64Image })
        });

        if (res.ok) {
            const data = await res.json();
            const isMirrored = (currentFacingMode === "user");
            renderDetections(data.faces, overlayCanvas, webcamVideo.videoWidth, webcamVideo.videoHeight, data.image_size, isMirrored);
            renderResultsList(data.faces);
            if (perfTime) perfTime.textContent = `${data.latency_ms} ms`;
        }
    } catch (err) {
        console.warn("Recognition cycle error:", err);
    } finally {
        isRequestInFlight = false;
    }
}

// ──────────────── Draw Bounding Boxes on Overlay Canvas ────────────────
function renderDetections(faces, canvas, srcW, srcH, backendSize = null, isMirrored = false) {
    if (!canvas) return;
    if (!srcW || !srcH) {
        srcW = canvas.width || 640;
        srcH = canvas.height || 480;
    }
    canvas.width = srcW;
    canvas.height = srcH;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!faces || faces.length === 0) return;

    // Hitung faktor skala antara resolusi inferensi backend dan resolusi tampilan canvas
    const backendW = (backendSize && backendSize.width) ? backendSize.width : srcW;
    const backendH = (backendSize && backendSize.height) ? backendSize.height : srcH;
    const scaleX = srcW / backendW;
    const scaleY = srcH / backendH;

    faces.forEach(face => {
        let w = face.box.w * scaleX;
        let h = face.box.h * scaleY;
        let x = face.box.x * scaleX;
        let y = face.box.y * scaleY;

        // Jika kamera depan (selfie) memiliki CSS scaleX(-1), balik koordinat horizontal agar pas di muka
        if (isMirrored) {
            x = srcW - (x + w);
        }

        const isOwner = face.is_owner;
        const isRecognized = face.is_recognized;

        // Styling based on identity
        let strokeColor = isOwner ? "#facc15" : (isRecognized ? "#10b981" : "#ef4444");
        let glowColor = isOwner ? "rgba(250, 204, 21, 0.6)" : (isRecognized ? "rgba(16, 185, 129, 0.5)" : "rgba(239, 68, 68, 0.5)");

        ctx.save();
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isOwner ? 20 : 12;
        ctx.lineWidth = isOwner ? 4 : 2.5;
        ctx.strokeStyle = strokeColor;

        // Draw Corner Brackets
        const len = Math.min(w, h) * 0.22;
        ctx.beginPath();
        // Top Left
        ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
        // Top Right
        ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
        // Bottom Right
        ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
        // Bottom Left
        ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
        ctx.stroke();

        // Label Tag
        const label = isOwner ? `👑 JOY MANALU • ${face.confidence}%` : `${face.name} • ${face.confidence}%`;
        ctx.font = isOwner ? "bold 15px 'Outfit', sans-serif" : "bold 13px 'Plus Jakarta Sans', sans-serif";
        const textWidth = ctx.measureText(label).width;
        const tagHeight = 26;
        const tagWidth = textWidth + 20;

        // Posisikan label di atas kotak wajah, atau di bawah jika terlalu dekat batas atas
        const tagY = (y >= tagHeight + 8) ? (y - tagHeight - 6) : Math.min(srcH - tagHeight, y + h + 6);
        // Jaga label tetap di dalam batas horizontal canvas
        const tagX = Math.max(0, Math.min(x, srcW - tagWidth));

        // Label Background
        ctx.fillStyle = isOwner ? "rgba(15, 20, 32, 0.9)" : "rgba(15, 20, 32, 0.85)";
        ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(tagX, tagY, tagWidth, tagHeight);

        // Label Text
        ctx.fillStyle = isOwner ? "#facc15" : "#ffffff";
        ctx.fillText(label, tagX + 10, tagY + 18);
        ctx.restore();
    });
}

// ──────────────── Results List Inspector Feed ────────────────
function renderResultsList(faces) {
    if (faceCountBadge) {
        faceCountBadge.textContent = `${faces.length} wajah`;
    }

    if (!faces || faces.length === 0) {
        resultsList.innerHTML = `
            <div class="empty-state">
                <div class="empty-reticle">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M8 15s1.5 2 4 2 4-2 4-2"/>
                        <line x1="9" y1="9" x2="9.01" y2="9"/>
                        <line x1="15" y1="9" x2="15.01" y2="9"/>
                    </svg>
                </div>
                <h4 class="empty-title">Menunggu Deteksi Wajah</h4>
                <p class="empty-desc">Nyalakan kamera atau upload foto untuk melihat hasil verifikasi identitas real-time.</p>
            </div>
        `;
        return;
    }

    let html = "";
    faces.forEach(face => {
        const isOwner = face.is_owner;
        const isRecognized = face.is_recognized;
        const cardClass = isOwner ? "face-card vip-owner" : (isRecognized ? "face-card recognized" : "face-card unknown");
        const initials = isOwner ? "👑" : (face.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase() || "?");
        const vipTagHtml = isOwner ? `<span class="vip-tag">👑 OWNER / VIP</span>` : "";

        html += `
            <div class="${cardClass}">
                <div class="face-avatar">${initials}</div>
                <div class="face-info">
                    <div class="face-name-row">
                        <span class="face-name">${face.name}</span>
                        ${vipTagHtml}
                    </div>
                    <div class="face-meta">
                        <span class="face-confidence">Skor: ${face.confidence}%</span>
                        <span class="face-distance">Dist: ${face.distance}</span>
                    </div>
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: ${face.confidence}%;"></div>
                </div>
            </div>
        `;
    });

    resultsList.innerHTML = html;
}

function clearResults() {
    renderResultsList([]);
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ──────────────── Photo Upload Mode ────────────────
function setupDropzone() {
    const uploadZone = document.getElementById("uploadZone");
    if (!uploadZone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove("drag-over");
        }, false);
    });

    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            processUploadFile(files[0]);
        }
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        processUploadFile(file);
    }
}

async function processUploadFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Silakan pilih file gambar JPG, PNG, atau WEBP.");
        return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = document.getElementById("uploadedImage");
        img.src = e.target.result;

        document.getElementById("uploadZone").classList.add("hidden");
        document.getElementById("uploadedPreview").classList.remove("hidden");

        if (processingIndicator) processingIndicator.classList.remove("hidden");

        img.onload = async () => {
            try {
                // Client-side resize before sending to keep response instant
                const tempCanvas = document.createElement("canvas");
                const scale = Math.min(1.0, 1080 / img.naturalWidth);
                tempCanvas.width = Math.round(img.naturalWidth * scale);
                tempCanvas.height = Math.round(img.naturalHeight * scale);

                const ctx = tempCanvas.getContext("2d");
                ctx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);

                const b64 = tempCanvas.toDataURL("image/jpeg", 0.85);

                const res = await fetch(`${API_BASE}/api/recognize`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: b64 })
                });

                if (res.ok) {
                    const data = await res.json();
                    renderDetections(data.faces, uploadOverlayCanvas, img.naturalWidth, img.naturalHeight, data.image_size, false);
                    renderResultsList(data.faces);
                    if (perfTime) perfTime.textContent = `${data.latency_ms} ms`;
                }
            } catch (err) {
                console.error("Upload error:", err);
                alert("Gagal memproses gambar: " + err.message);
            } finally {
                if (processingIndicator) processingIndicator.classList.add("hidden");
            }
        };
    };
    reader.readAsDataURL(file);
}

function removeUpload() {
    document.getElementById("uploadedImage").src = "";
    document.getElementById("uploadZone").classList.remove("hidden");
    document.getElementById("uploadedPreview").classList.add("hidden");
    document.getElementById("fileInput").value = "";
    const ctx = uploadOverlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, uploadOverlayCanvas.width, uploadOverlayCanvas.height);
    clearResults();
}

// ──────────────── Identity Vault Database & Kartu 2 Wajah Terdaftar ────────────────
async function loadPersonsVault() {
    try {
        const res = await fetch(`${API_BASE}/api/persons`);
        if (res.ok) {
            personsDatabase = await res.json();
            renderRegisteredList(personsDatabase);
            renderPersonsGrid(personsDatabase);
            if (statPersons) statPersons.textContent = personsDatabase.length;
            const counter = document.getElementById("registeredCounter");
            if (counter) counter.textContent = `${personsDatabase.length} orang`;
        }
    } catch (err) {
        console.warn("Could not load persons database:", err);
        const listEl = document.getElementById("registeredPersonsList");
        if (listEl) {
            listEl.innerHTML = `<div class="list-empty-msg">Gagal memuat data wajah.</div>`;
        }
        if (personsGrid) {
            personsGrid.innerHTML = `<div class="loading-state">Gagal memuat database identitas. Pastikan server backend aktif.</div>`;
        }
    }
}

// Render KARTU 2: "Wajah terdaftar"
function renderRegisteredList(persons) {
    const listEl = document.getElementById("registeredPersonsList");
    if (!listEl) return;

    if (!persons || persons.length === 0) {
        listEl.innerHTML = `<div class="list-empty-msg">Belum ada wajah terdaftar.</div>`;
        return;
    }

    let html = "";
    persons.forEach(p => {
        const isOwner = p.is_owner;
        const count = p.sample_count !== undefined ? p.sample_count : (p.embedded_count || p.photo_count || 0);
        const lowWarning = count < 30 ? `<span class="sample-warning-low"> &middot; kurang dari 30</span>` : "";
        const crown = isOwner ? `<span class="vip-crown-icon" title="VIP Owner">👑</span>` : "";

        // Escape for attributes
        const safeName = p.name.replace(/"/g, '&quot;');
        const safeId = p.id.replace(/"/g, '&quot;');

        html += `
            <div class="person-row-item" data-id="${safeId}">
                <div class="person-row-info">
                    <div class="person-row-name-wrap">
                        <span class="person-row-name" onclick="renamePerson('${safeId}', '${safeName}')" title="Ketuk untuk mengubah nama">
                            ${p.name}
                        </span>
                        ${crown}
                    </div>
                    <div class="person-row-samples">
                        ${count} sampel${lowWarning}
                    </div>
                </div>
                <div class="person-row-actions">
                    <button type="button" class="btn-add-sample-mini" onclick="addSampleToPerson('${safeId}', '${safeName}')" title="Tambah sampel untuk orang ini">
                        + Sampel
                    </button>
                    <button type="button" class="btn-delete-trash" onclick="deletePerson('${safeId}', '${safeName}')" title="Hapus data ${safeName}">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            <line x1="10" y1="11" x2="10" y2="17"/>
                            <line x1="14" y1="11" x2="14" y2="17"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    });

    listEl.innerHTML = html;
}

// ──────────────── KARTU 1: Pendaftaran Wajah ────────────────
let targetPersonForSample = null;

function showRegFeedback(msg, type = "success") {
    const el = document.getElementById("regFeedback");
    if (!el) return;
    el.textContent = msg;
    el.className = `reg-feedback ${type}`;
    el.classList.remove("hidden");
    setTimeout(() => {
        el.classList.add("hidden");
    }, 4500);
}

// 1. Rekam langsung dari kamera
async function triggerRecordFace() {
    let name = document.getElementById("regNameInput") ? document.getElementById("regNameInput").value.trim() : "";
    if (targetPersonForSample) {
        name = targetPersonForSample;
    }
    if (!name) {
        alert("Silakan ketik nama terlebih dahulu di kolom nama (misal: Joy Manalu).");
        if (document.getElementById("regNameInput")) {
            document.getElementById("regNameInput").focus();
        }
        return;
    }

    // Pastikan webcam berjalan
    if (!webcamStream || !webcamVideo || !webcamVideo.videoWidth) {
        showRegFeedback("Menyalakan kamera...", "success");
        try {
            await startWebcam();
            // Beri waktu 800ms agar stream video aktif
            await new Promise(r => setTimeout(r, 800));
        } catch (e) {
            showRegFeedback("Kamera belum diizinkan atau tidak aktif.", "error");
            return;
        }
    }

    if (!webcamVideo || !webcamVideo.videoWidth) {
        showRegFeedback("Kamera belum siap, coba klik 'Rekam' sekali lagi.", "error");
        return;
    }

    showRegFeedback("Merekam wajah & mengekstrak vektor 512D...", "success");

    try {
        const canvas = document.createElement("canvas");
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        const b64 = canvas.toDataURL("image/jpeg", 0.90);

        const res = await fetch(`${API_BASE}/api/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, image: b64 })
        });

        const data = await res.json();
        if (data.success) {
            showRegFeedback(`✓ Berhasil merekam sampel untuk ${data.name}! Total: ${data.total_samples} sampel.`, "success");
            targetPersonForSample = null;
            await loadPersonsVault();
            checkServerHealth();
        } else {
            showRegFeedback(`Gagal: ${data.error || "Wajah tidak terdeteksi di kamera"}`, "error");
        }
    } catch (e) {
        showRegFeedback(`Error rekam: ${e.message}`, "error");
    }
}

// 2. Pilih foto
async function handleRegisterFiles(e) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    let name = document.getElementById("regNameInput") ? document.getElementById("regNameInput").value.trim() : "";
    if (targetPersonForSample) {
        name = targetPersonForSample;
    }
    if (!name) {
        name = prompt("Masukkan nama untuk foto yang dipilih:");
        if (!name || !name.trim()) {
            e.target.value = "";
            return;
        }
        name = name.trim();
        if (document.getElementById("regNameInput")) document.getElementById("regNameInput").value = name;
    }

    showRegFeedback(`Memproses ${files.length} foto untuk ${name}...`, "success");

    try {
        const formData = new FormData();
        formData.append("name", name);
        files.forEach(f => formData.append("images", f));

        const res = await fetch(`${API_BASE}/api/register`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (data.success) {
            showRegFeedback(`✓ Berhasil menambahkan sampel untuk ${data.name}! Total: ${data.total_samples} sampel.`, "success");
            targetPersonForSample = null;
            await loadPersonsVault();
            checkServerHealth();
        } else {
            showRegFeedback(`Gagal: ${data.error || "Tidak ada wajah valid"}`, "error");
        }
    } catch (err) {
        showRegFeedback(`Error: ${err.message}`, "error");
    } finally {
        e.target.value = "";
    }
}

function openFolderPicker() {
    const el = document.getElementById('regFolderInput');
    if (el) {
        el.value = "";
        el.click();
    }
}

// 3. Pilih folder (banyak orang sekaligus)
async function handleRegisterFolder(e) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    // Filter file gambar saja
    const validFiles = files.filter(f => {
        const name = f.name.toLowerCase();
        return name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || name.endsWith('.webp');
    });

    if (validFiles.length === 0) {
        alert("Tidak ada file foto (JPG, PNG, WEBP) di dalam folder yang dipilih.");
        e.target.value = "";
        return;
    }

    // Kelompokkan per subfolder (nama orang)
    const groups = {};
    for (const f of validFiles) {
        const rel = f.webkitRelativePath || f.name;
        const parts = rel.split("/");
        let personName = "";
        if (parts.length >= 3) {
            // Contoh: "DatasetUtama/Joy Manalu/01.jpg"
            personName = parts[parts.length - 2];
        } else if (parts.length === 2) {
            // Contoh: "Joy Manalu/01.jpg"
            personName = parts[0];
        } else {
            // Jika memilih file langsung tanpa subfolder, pakai nama di input nama atau 'Wajah Baru'
            const typedName = document.getElementById("regNameInput") ? document.getElementById("regNameInput").value.trim() : "";
            personName = typedName || "Wajah Baru";
        }

        // Bersihkan nama dari akhiran timestamp Google Drive (misal: "Devi Manurung -20261019T050317Z-1-001")
        let cleanName = personName.replace(/\s*-\d{8}T\d{6}Z.*$/i, "").replace(/[-_]/g, " ").trim();
        if (!cleanName) cleanName = personName;

        if (!groups[cleanName]) groups[cleanName] = [];
        groups[cleanName].push(f);
    }

    const folderNames = Object.keys(groups);
    if (folderNames.length === 0) {
        alert("Format folder tidak sesuai.");
        e.target.value = "";
        return;
    }

    const confirmMsg = `Ditemukan ${folderNames.length} identitas (${validFiles.length} foto):\n\n` +
        folderNames.slice(0, 8).map(n => `• ${n} (${groups[n].length} foto)`).join("\n") +
        (folderNames.length > 8 ? `\n... dan ${folderNames.length - 8} orang lainnya.` : "") +
        `\n\nDaftarkan semuanya sekarang?`;

    if (!confirm(confirmMsg)) {
        e.target.value = "";
        return;
    }

    showRegFeedback(`Mendaftarkan 0/${folderNames.length} identitas...`, "success");

    let successCount = 0;
    let totalSamplesAdded = 0;

    for (let i = 0; i < folderNames.length; i++) {
        const fName = folderNames[i];
        showRegFeedback(`Mendaftarkan (${i + 1}/${folderNames.length}): ${fName}...`, "success");

        try {
            // Kirim per batch maksimal 10 foto per request agar tidak timeout
            const personFiles = groups[fName];
            const batchSize = 10;
            for (let b = 0; b < personFiles.length; b += batchSize) {
                const chunk = personFiles.slice(b, b + batchSize);
                const formData = new FormData();
                formData.append("name", fName);
                chunk.forEach(f => formData.append("images", f));

                const res = await fetch(`${API_BASE}/api/register`, {
                    method: "POST",
                    body: formData
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.success) {
                        totalSamplesAdded += (data.total_samples || chunk.length);
                    }
                }
            }
            successCount++;
        } catch (err) {
            console.error(`Gagal upload untuk ${fName}:`, err);
        }
    }

    showRegFeedback(`✓ Selesai! Berhasil mendaftarkan ${successCount}/${folderNames.length} identitas.`, "success");
    e.target.value = "";
    await loadPersonsVault();
    checkServerHealth();
}

// ──────────────── TAMBAHAN: GANTI NAMA & GABUNGKAN ────────────────
async function renamePerson(personId, currentName) {
    const newName = prompt(`Ubah nama untuk "${currentName}":`, currentName);
    if (!newName || !newName.trim() || newName.trim() === currentName) {
        return;
    }

    const cleanNewName = newName.trim();

    try {
        const res = await fetch(`${API_BASE}/api/person/rename`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ person_id: personId, new_name: cleanNewName })
        });

        const data = await res.json();

        // Cek jika nama sama dengan orang lain -> Tanyakan "Gabungkan data?"
        if (!data.success && data.can_merge) {
            const doMerge = confirm(`Gabungkan data?\n\nNama "${cleanNewName}" sudah ada dalam daftar. Ingin menyatukan semua sampel "${currentName}" ke dalam "${cleanNewName}"?`);
            if (doMerge) {
                const mergeRes = await fetch(`${API_BASE}/api/person/merge`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source_id: personId,
                        target_id: data.existing_id,
                        target_name: cleanNewName
                    })
                });
                const mergeData = await mergeRes.json();
                if (mergeData.success) {
                    alert(`✓ ${mergeData.message}`);
                    await loadPersonsVault();
                    checkServerHealth();
                } else {
                    alert(`Gagal menggabungkan: ${mergeData.error}`);
                }
            }
            return;
        }

        if (data.success) {
            alert(`✓ ${data.message}`);
            await loadPersonsVault();
            checkServerHealth();
        } else {
            alert(`Gagal: ${data.error}`);
        }
    } catch (e) {
        alert(`Error koneksi: ${e.message}`);
    }
}

// ──────────────── TAMBAHAN: TAMBAH SAMPEL ────────────────
function addSampleToPerson(personId, personName) {
    targetPersonForSample = personName;
    const choice = confirm(`Tambah sampel untuk "${personName}"?\n\n• Klik [OK] untuk Rekam dari Kamera langsung\n• Klik [Batal] untuk Pilih file foto dari galeri/perangkat`);
    if (choice) {
        // Rekam kamera
        triggerRecordFace();
    } else {
        // Pilih file foto
        document.getElementById('regFileInput').click();
    }
}

// ──────────────── TAMBAHAN: HAPUS DENGAN KONFIRMASI ────────────────
async function deletePerson(personId, personName) {
    const confirmed = confirm(`Hapus data ${personName}?`);
    if (!confirmed) return;

    try {
        const res = await fetch(`${API_BASE}/api/person/${encodeURIComponent(personId)}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (data.success) {
            showRegFeedback(`✓ ${data.message}`, "success");
            await loadPersonsVault();
            checkServerHealth();
        } else {
            alert(`Gagal menghapus: ${data.error}`);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

// ──────────────── TAMBAHAN: TOMBOL KALIBRASI ────────────────
async function triggerCalibrateThreshold() {
    const btn = document.getElementById("btnCalibrate");
    const statusTxt = document.getElementById("calibStatusTxt");
    if (btn) btn.disabled = true;
    if (statusTxt) statusTxt.textContent = "Menghitung...";

    try {
        const res = await fetch(`${API_BASE}/api/calibrate-auto`, {
            method: "POST"
        });
        const data = await res.json();
        if (data.success) {
            if (statusTxt) statusTxt.textContent = "Kalibrasi selesai";
            alert(`✓ Kalibrasi selesai!\nAmbang kecocokan optimal: ${data.threshold}\nTotal sampel dihitung: ${data.total_samples}`);
            setTimeout(() => {
                if (statusTxt) statusTxt.textContent = "";
            }, 5000);
        } else {
            if (statusTxt) statusTxt.textContent = "Gagal";
            alert(`Kalibrasi gagal: ${data.error}`);
        }
    } catch (e) {
        if (statusTxt) statusTxt.textContent = "Error";
        alert(`Error: ${e.message}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

function renderPersonsGrid(persons) {
    if (!personsGrid) return;

    if (!persons || persons.length === 0) {
        personsGrid.innerHTML = `<div class="loading-state">Tidak ada identitas yang ditemukan.</div>`;
        return;
    }

    let html = "";
    persons.forEach(p => {
        const isOwner = p.is_owner;
        const chipClass = isOwner ? "person-chip vip-owner" : "person-chip";
        const avatar = isOwner ? "👑" : (p.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase() || "ID");
        const verifiedIcon = isOwner ? "★" : "✓";
        const count = p.sample_count !== undefined ? p.sample_count : (p.embedded_count || p.photo_count || 0);

        html += `
            <div class="${chipClass}" title="${p.name} (${count} foto)">
                <div class="person-avatar">${avatar}</div>
                <div class="person-info-col">
                    <span class="person-name">${p.name}</span>
                    <span class="person-photos-tag">${count} Foto Terverifikasi</span>
                </div>
                <div class="person-verified-icon">${verifiedIcon}</div>
            </div>
        `;
    });

    personsGrid.innerHTML = html;
}

function filterPersonsList(query) {
    const q = query.toLowerCase().trim();
    if (!q) {
        renderRegisteredList(personsDatabase);
        renderPersonsGrid(personsDatabase);
        return;
    }
    const filtered = personsDatabase.filter(p => p.name.toLowerCase().includes(q));
    renderRegisteredList(filtered);
    renderPersonsGrid(filtered);
}

// ──────────────── Mobile Connect Modal & QR Code ────────────────
async function openMobileModal() {
    mobileModal.classList.remove("hidden");
    
    // Fetch latest network info for local Wi-Fi URL
    try {
        const res = await fetch(`${API_BASE}/api/network-info`);
        if (res.ok) {
            const data = await res.json();
            const targetUrl = data.tunnel_url || data.http_url || mobileUrlInput.value;
            mobileUrlInput.value = targetUrl;
            drawQRCode(targetUrl);
        }
    } catch (e) {
        drawQRCode(mobileUrlInput.value);
    }
}

function closeMobileModal(e) {
    if (e && e.target !== mobileModal && !e.target.classList.contains("modal-close-btn") && !e.target.classList.contains("btn-modal-primary")) {
        return;
    }
    mobileModal.classList.add("hidden");
}

function copyMobileUrl() {
    mobileUrlInput.select();
    navigator.clipboard.writeText(mobileUrlInput.value);
    const btn = document.querySelector(".btn-copy");
    const original = btn.textContent;
    btn.textContent = "✓ Tersalin!";
    setTimeout(() => { btn.textContent = original; }, 2000);
}

// Canvas-based QR Code Renderer (Lightweight & Standalone)
function drawQRCode(url) {
    const canvas = document.getElementById("qrCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    
    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);

    // Draw Stylized Cyber QR Mockup with High Contrast Pattern & Corner Markers
    const cellSize = Math.floor(size / 25);
    ctx.fillStyle = "#0a0d14";

    // Corner Finder Patterns
    drawFinderPattern(ctx, 2 * cellSize, 2 * cellSize, cellSize * 7);
    drawFinderPattern(ctx, (25 - 9) * cellSize, 2 * cellSize, cellSize * 7);
    drawFinderPattern(ctx, 2 * cellSize, (25 - 9) * cellSize, cellSize * 7);

    // Algorithmic pattern generator based on URL hash
    let hash = 0;
    for (let i = 0; i < url.length; i++) {
        hash = ((hash << 5) - hash) + url.charCodeAt(i);
        hash |= 0;
    }

    for (let row = 0; row < 25; row++) {
        for (let col = 0; col < 25; col++) {
            // Skip finder patterns
            if ((row < 9 && col < 9) || (row < 9 && col > 15) || (row > 15 && col < 9)) {
                continue;
            }
            const bit = (Math.sin(row * 13 + col * 17 + hash) > 0.1);
            if (bit) {
                ctx.fillRect(col * cellSize, row * cellSize, cellSize - 0.5, cellSize - 0.5);
            }
        }
    }
}

function drawFinderPattern(ctx, x, y, s) {
    ctx.fillStyle = "#0a0d14";
    ctx.fillRect(x, y, s, s);
    ctx.fillStyle = "#ffffff";
    const sub = s / 7;
    ctx.fillRect(x + sub, y + sub, s - 2 * sub, s - 2 * sub);
    ctx.fillStyle = "#06b6d4";
    ctx.fillRect(x + 2 * sub, y + 2 * sub, s - 4 * sub, s - 4 * sub);
}
