import { FormEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import ImageModal from "../components/ImageModal";
import "./MeasurePage.css";

type Point = { x: number; y: number };

type MeasureResponse = {
  ok: boolean;
  length_m: number;
  length_cm: number;
  pixel_distance: number;
  point1: Point;
  point2: Point;
  annotated_url: string;
  actual_length_cm?: number;
  actual_length_m?: number;
  absolute_error_cm?: number;
  absolute_error_m?: number;
  relative_error_pct?: number;
};

const DEFAULT_FX = 4172.88879;
const DEFAULT_FY = 4164.42586;
const DEFAULT_DISTANCE_Z = 0.264;

const toAbsolute = (path: string) => {
  if (!path) return path;
  try {
    return path.startsWith("http") ? path : new URL(path, api.defaults.baseURL).toString();
  } catch {
    return path;
  }
};

export function MeasurePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [points, setPoints] = useState<Point[]>([]);

  const [fx, setFx] = useState(DEFAULT_FX);
  const [fy, setFy] = useState(DEFAULT_FY);
  const [distanceZ, setDistanceZ] = useState(DEFAULT_DISTANCE_Z);
  const [actualDistance, setActualDistance] = useState<number | "">("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MeasureResponse | null>(null);
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const fileSummary = useMemo(() => {
    if (!file) return "No image selected.";
    return `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  }, [file]);

  const handleFileChange = (selected: File | null) => {
    setFile(selected);
    setPoints([]);
    setResult(null);
    setNaturalSize(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (selected) {
      const nextUrl = URL.createObjectURL(selected);
      setPreviewUrl(nextUrl);
    } else {
      setPreviewUrl(null);
    }
  };

  const handlePreviewClick = (event: MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !naturalSize) return;
    if (!file) return;
    const rect = imageRef.current.getBoundingClientRect();
    const relX = (event.clientX - rect.left) / rect.width;
    const relY = (event.clientY - rect.top) / rect.height;
    if (relX < 0 || relY < 0 || relX > 1 || relY > 1) return;
    const absPoint: Point = {
      x: relX * naturalSize.width,
      y: relY * naturalSize.height,
    };
    setPoints((prev) => {
      if (prev.length === 2) return [absPoint];
      return [...prev, absPoint];
    });
    setError(null);
  };

  const handleSubmit = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    if (!file) {
      setError("Please upload an image to measure.");
      return;
    }
    if (points.length !== 2) {
      setError("Select exactly two points on the image.");
      return;
    }
    const formData = new FormData();
    formData.append("image", file);
    formData.append("fx", String(fx));
    formData.append("fy", String(fy));
    formData.append("distance_z", String(distanceZ));
    formData.append("points", JSON.stringify(points));
    if (actualDistance !== "" && !Number.isNaN(actualDistance)) {
      formData.append("actual_cm", String(actualDistance));
    }
    try {
      setLoading(true);
      setError(null);
      const { data } = await api.post<MeasureResponse>("/api/measure/distance", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult({
        ...data,
        annotated_url: toAbsolute(data.annotated_url),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute measurement.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const clearPoints = () => {
    setPoints([]);
    setResult(null);
  };

  const getPointStyle = (pt: Point) => {
    if (!naturalSize) return {};
    return {
      left: `${(pt.x / naturalSize.width) * 100}%`,
      top: `${(pt.y / naturalSize.height) * 100}%`,
    };
  };

  return (
    <section className="measure-root">
      <header className="measure-header">
        <p className="eyebrow">Module 1 · Measurement Lab</p>
        <h1>Measure Real-World Distances</h1>
        <p>
          Upload a calibrated capture, click the two endpoints, and let the backend convert pixel gaps into metric distances using your
          camera intrinsics.
        </p>
      </header>

      <div className="measure-grid">
        <form className="measure-panel" onSubmit={handleSubmit}>
          <label className="field">
            <span>Measurement image</span>
            <input type="file" accept="image/*" onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)} />
            <small>{fileSummary}</small>
          </label>

          <div className="field-grid">
            <label className="field">
              <span>fx</span>
              <input type="number" step={0.01} value={fx} onChange={(event) => setFx(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>fy</span>
              <input type="number" step={0.01} value={fy} onChange={(event) => setFy(Number(event.target.value))} />
            </label>
          </div>

          <label className="field">
            <span>Camera–object distance Z (meters)</span>
            <input type="number" step={0.001} value={distanceZ} onChange={(event) => setDistanceZ(Number(event.target.value))} />
          </label>

          <label className="field">
            <span>Actual distance (cm)</span>
            <input
              type="number"
              step={0.01}
              placeholder="Enter ground-truth length"
              value={actualDistance}
              onChange={(event) => {
                const value = event.target.value;
                setActualDistance(value === "" ? "" : Number(value));
              }}
            />
            <small className="field-hint">Optional: add a real-world measurement to compute error and accuracy.</small>
          </label>

          <div className="field-row spaced">
            <p className="field-hint">Click twice on the preview to capture endpoints. Use reset if you need to start over.</p>
            <button type="button" className="ghost-btn" onClick={clearPoints} disabled={!points.length}>
              Reset points
            </button>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="primary-btn" disabled={loading}>
            {loading ? "Measuring…" : "Compute distance"}
          </button>
        </form>

        <div className="measure-preview-card">
          <h2>Image Preview</h2>
          <div className="measure-preview" onClick={handlePreviewClick}>
            {previewUrl ? (
              <>
                <img
                  ref={imageRef}
                  src={previewUrl}
                  alt="Measurement preview"
                  onLoad={(event) => {
                    setNaturalSize({
                      width: event.currentTarget.naturalWidth,
                      height: event.currentTarget.naturalHeight,
                    });
                  }}
                />
                {naturalSize && (
                  <div className="measure-overlay">
                    {points.map((pt, idx) => (
                      <span key={`${pt.x}-${pt.y}-${idx}`} className="measure-point" style={getPointStyle(pt)}>
                        {idx + 1}
                      </span>
                    ))}
                    {points.length === 2 && (
                      <svg
                        className="measure-line"
                        viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
                        preserveAspectRatio="none"
                      >
                        <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} />
                      </svg>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="placeholder">Upload an image to begin marking reference points.</p>
            )}
          </div>
          <p className="preview-hint">Click directly on the preview to place markers. The second click finalizes the segment.</p>
        </div>

        <div className="measure-results">
          <h2>Results</h2>
          {!result && <p className="placeholder">Submit the form to calculate the metric distance and annotated overlay.</p>}

          {result && (
            <>
              <div className="metrics-grid">
                <div>
                  <p className="metric-label">Length (cm)</p>
                  <p className="metric-value">{result.length_cm.toFixed(2)} cm</p>
                </div>
                <div>
                  <p className="metric-label">Length (m)</p>
                  <p className="metric-value">{result.length_m.toFixed(4)} m</p>
                </div>
                <div>
                  <p className="metric-label">Pixel distance</p>
                  <p className="metric-value">{result.pixel_distance.toFixed(2)} px</p>
                </div>
              </div>
              <div className="points-list">
                <p>
                  P1: ({result.point1.x.toFixed(1)}, {result.point1.y.toFixed(1)})
                </p>
                <p>
                  P2: ({result.point2.x.toFixed(1)}, {result.point2.y.toFixed(1)})
                </p>
              </div>
              {result.actual_length_cm !== undefined && (
                <div className="metrics-grid compact">
                  <div>
                    <p className="metric-label">Actual (cm)</p>
                    <p className="metric-value">{result.actual_length_cm.toFixed(2)} cm</p>
                  </div>
                  <div>
                    <p className="metric-label">Abs. error</p>
                    <p className="metric-value">{result.absolute_error_cm?.toFixed(2)} cm</p>
                  </div>
                  {typeof result.relative_error_pct === "number" && (
                    <div>
                      <p className="metric-label">Accuracy</p>
                      <p className="metric-value">
                        {(100 - result.relative_error_pct).toFixed(2)}
                        <span className="metric-suffix">%</span>
                      </p>
                    </div>
                  )}
                </div>
              )}
              {result.annotated_url && (
                <div className="annotated-card">
                  <img
                    src={result.annotated_url}
                    alt="Annotated measurement"
                    role="button"
                    tabIndex={0}
                    onClick={() => setModalImage({ src: result.annotated_url, title: "Annotated measurement" })}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") setModalImage({ src: result.annotated_url, title: "Annotated measurement" });
                    }}
                  />
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={() => setModalImage({ src: result.annotated_url, title: "Annotated measurement" })}
                  >
                    Expand preview
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <ImageModal open={Boolean(modalImage)} onClose={() => setModalImage(null)} title={modalImage?.title}>
        {modalImage && <img src={modalImage.src} alt={modalImage.title} />}
      </ImageModal>
    </section>
  );
}

export default MeasurePage;
