import { useState, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [confidence, setConfidence] = useState(0.35);
  const [iou, setIou] = useState(0.45);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  /* ---- File handling ---- */
  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!['image/jpeg', 'image/png'].includes(f.type)) {
      setError('Only JPG and PNG images are accepted.');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  }, [handleFile]);

  const removeFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  /* ---- Detect ---- */
  const runDetection = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const url = `${API_BASE}/detect?confidence=${confidence}&iou=${iou}`;
      const res = await fetch(url, { method: 'POST', body: formData });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Detection failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  /* ---- Downloads ---- */
  const downloadImage = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.annotated_image_b64}`;
    link.download = `headcount_${result.id}.png`;
    link.click();
  };

  const downloadJSON = () => {
    if (!result) return;
    window.open(`${API_BASE}/download/json/${result.id}`, '_blank');
  };

  const downloadCSV = () => {
    if (!result) return;
    window.open(`${API_BASE}/download/csv/${result.id}`, '_blank');
  };

  /* ---- Helpers ---- */
  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  const formatTime = (iso) => {
    return new Date(iso).toLocaleString();
  };

  /* ---- Render ---- */
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">👁</div>
            <span>HeadCount</span>
          </div>
          <span className="header-badge">YOLOv8 Powered</span>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <h1>Classroom Student Counter</h1>
        <p>
          Upload a classroom photo and let AI count every student — even those
          partially hidden behind desks.
        </p>
      </section>

      {/* Main */}
      <main className="main-content">
        {/* Upload zone */}
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <span className="upload-icon">📷</span>
          <h3>Drop your classroom image here</h3>
          <p>or click to browse · JPG, PNG accepted</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {/* Preview strip */}
        {file && (
          <div className="preview-strip">
            <img className="preview-thumb" src={preview} alt="Preview" />
            <div className="preview-info">
              <div className="filename">{file.name}</div>
              <div className="filesize">{formatBytes(file.size)}</div>
            </div>
            <button className="btn-remove" onClick={removeFile} title="Remove">
              ✕
            </button>
          </div>
        )}

        {/* Controls */}
        {file && (
          <div className="controls-panel">
            <div className="slider-group">
              <div className="slider-label">
                <span>Confidence Threshold</span>
                <span className="slider-value">{confidence.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1"
                step="0.01"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
              />
            </div>

            <div className="slider-group">
              <div className="slider-label">
                <span>IoU / NMS Threshold</span>
                <span className="slider-value">{iou.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1"
                step="0.01"
                value={iou}
                onChange={(e) => setIou(parseFloat(e.target.value))}
              />
            </div>

            <button
              className="btn-detect"
              onClick={runDetection}
              disabled={loading || !file}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  Detecting…
                </>
              ) : (
                <>🔍 Count Students</>
              )}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="error-toast">
            ⚠️ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="results-section">
            {/* Stats */}
            <div className="stats-row">
              <div className="stat-card primary">
                <div className="stat-number">{result.student_count}</div>
                <div className="stat-label">Students Detected</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{result.confidence_threshold}</div>
                <div className="stat-label">Confidence</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{result.iou_threshold}</div>
                <div className="stat-label">IoU Threshold</div>
              </div>
              <div className="stat-card">
                <div className="stat-number" style={{ fontSize: '0.95rem', lineHeight: '2.8rem' }}>
                  {formatTime(result.timestamp)}
                </div>
                <div className="stat-label">Timestamp</div>
              </div>
            </div>

            {/* Annotated image + export */}
            <div className="result-image-container">
              <img
                src={`data:image/png;base64,${result.annotated_image_b64}`}
                alt="Annotated classroom"
              />
              <div className="export-bar">
                <button className="btn-export" onClick={downloadImage}>
                  🖼️ Download Image
                </button>
                <button className="btn-export" onClick={downloadJSON}>
                  📄 Export JSON
                </button>
                <button className="btn-export" onClick={downloadCSV}>
                  📊 Export CSV
                </button>
                <div className="export-spacer" />
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', alignSelf: 'center' }}>
                  Session: {result.id}
                </span>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        HeadCount v1.0 — Built with FastAPI + YOLOv8 + React
      </footer>
    </div>
  );
}

export default App;
