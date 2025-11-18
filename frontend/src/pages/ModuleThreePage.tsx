import { FormEvent, useMemo, useState } from "react";
import { api } from "../api/client";
import "./ModuleThreePage.css";

type GradientResult = {
  image: string;
  mag_url: string;
  angle_url: string;
  log_url: string;
  grid_url: string;
  edge_density: number;
  log_energy: number;
};

type FeatureResult = {
  image: string;
  edges_binary_url: string;
  edges_overlay_url: string;
  edge_keypoints_url: string;
  harris_response_url: string;
  corner_overlay_url: string;
  edge_pixel_count: number;
  edge_keypoint_count: number;
  corner_count: number;
};

type BoundaryResult = {
  image: string;
  found: boolean;
  edges_url: string;
  edges_closed_url: string;
  bbox_overlay_url: string;
  score?: number;
  rectangularity?: number;
  center_score?: number;
  area_fraction?: number;
  contour_area?: number;
  box_width?: number;
  box_height?: number;
  angle?: number;
};

const toAbsolute = (path: string) => {
  if (!path) return path;
  try {
    return path.startsWith("http") ? path : new URL(path, api.defaults.baseURL).toString();
  } catch {
    return path;
  }
};

export function ModuleThreePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [sigma, setSigma] = useState(1.4);
  const [logKsize, setLogKsize] = useState(3);
  const [edgeSigma, setEdgeSigma] = useState(1.0);
  const [lowThresh, setLowThresh] = useState(20);
  const [highThresh, setHighThresh] = useState(60);
  const [cornerThresh, setCornerThresh] = useState(0.02);
  const [nmsRadius, setNmsRadius] = useState(6);
  const [edgeStride, setEdgeStride] = useState(5);

  const [gradResults, setGradResults] = useState<GradientResult[] | null>(null);
  const [featureResults, setFeatureResults] = useState<FeatureResult[] | null>(null);
  const [boundaryResults, setBoundaryResults] = useState<BoundaryResult[] | null>(null);

  const [gradLoading, setGradLoading] = useState(false);
  const [featureLoading, setFeatureLoading] = useState(false);
  const [boundaryLoading, setBoundaryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileSummary = useMemo(() => {
    if (!files.length) return "No images selected yet.";
    return `${files.length} image${files.length > 1 ? "s" : ""} selected.`;
  }, [files]);

  const handleFiles = (list: FileList | null) => {
    if (!list) {
      setFiles([]);
      return;
    }
    const imgs = Array.from(list).filter((file) => file.type.startsWith("image/"));
    setFiles(imgs);
  };

  const requireFiles = () => {
    if (!files.length) {
      setError("Please upload at least one object image.");
      return false;
    }
    setError(null);
    return true;
  };

  const runGradients = async () => {
    if (!requireFiles()) return;
    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));
    formData.append("sigma", String(sigma));
    formData.append("ksize", String(logKsize));
    try {
      setGradLoading(true);
      const { data } = await api.post<{ results: GradientResult[] }>("/api/edge/gradients", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setGradResults(
        data.results.map((item) => ({
          ...item,
          mag_url: toAbsolute(item.mag_url),
          angle_url: toAbsolute(item.angle_url),
          log_url: toAbsolute(item.log_url),
          grid_url: toAbsolute(item.grid_url),
        }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute gradients.");
    } finally {
      setGradLoading(false);
    }
  };

  const runFeatures = async () => {
    if (!requireFiles()) return;
    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));
    formData.append("sigma", String(edgeSigma));
    formData.append("low", String(lowThresh));
    formData.append("high", String(highThresh));
    formData.append("corner_thresh", String(cornerThresh));
    formData.append("nms_radius", String(nmsRadius));
    formData.append("edge_stride", String(edgeStride));
    try {
      setFeatureLoading(true);
      const { data } = await api.post<{ results: FeatureResult[] }>("/api/edge/features", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setFeatureResults(
        data.results.map((item) => ({
          ...item,
          edges_binary_url: toAbsolute(item.edges_binary_url),
          edges_overlay_url: toAbsolute(item.edges_overlay_url),
          edge_keypoints_url: toAbsolute(item.edge_keypoints_url),
          harris_response_url: toAbsolute(item.harris_response_url),
          corner_overlay_url: toAbsolute(item.corner_overlay_url),
        }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run edge/corner analysis.");
    } finally {
      setFeatureLoading(false);
    }
  };

  const runBoundaries = async () => {
    if (!requireFiles()) return;
    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));
    try {
      setBoundaryLoading(true);
      const { data } = await api.post<{ results: BoundaryResult[] }>("/api/edge/boundaries", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setBoundaryResults(
        data.results.map((item) => ({
          ...item,
          edges_url: toAbsolute(item.edges_url),
          edges_closed_url: toAbsolute(item.edges_closed_url),
          bbox_overlay_url: toAbsolute(item.bbox_overlay_url),
        }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to extract boundaries.");
    } finally {
      setBoundaryLoading(false);
    }
  };

  return (
    <section className="module3-root">
      <header className="module3-header">
        <p className="eyebrow">Module 3 · Dataset Lab</p>
        <h1>Edge & Boundary Analysis</h1>
        <p>
          Upload a small dataset (≈10 images of your measurement object). Run gradient + LoG diagnostics, inspect adaptive edge/corner
          detections, and extract final object boundaries—all powered by your Module 3 scripts.
        </p>
      </header>

      <div className="upload-panel">
        <label className="field">
          <span>Dataset images</span>
          <input type="file" accept="image/*" multiple onChange={(event) => handleFiles(event.target.files)} />
          <small>{fileSummary}</small>
        </label>
      </div>

      {error && <p className="form-error">{error}</p>}

      <div className="module3-grid">
        <article>
          <header>
            <h2>1. Gradients & LoG</h2>
            <p>Visualize magnitude, direction, and Laplacian-of-Gaussian responses for every frame in your dataset.</p>
          </header>
          <form
            className="mini-form"
            onSubmit={(evt: FormEvent) => {
              evt.preventDefault();
              runGradients();
            }}
          >
            <label>
              Sigma
              <input type="number" step={0.1} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
            </label>
            <label>
              LoG kernel size
              <input type="number" min={1} step={2} value={logKsize} onChange={(e) => setLogKsize(Number(e.target.value))} />
            </label>
            <button type="submit" className="primary-btn" disabled={gradLoading}>
              {gradLoading ? "Processing…" : "Run Gradients"}
            </button>
          </form>
          {gradResults && (
            <div className="result-list">
              {gradResults.map((item) => (
                <div key={item.image} className="result-card">
                  <h3>{item.image}</h3>
                  <p>Edge density: {(item.edge_density * 100).toFixed(1)}%</p>
                  <p>LoG energy: {item.log_energy.toFixed(4)}</p>
                  <div className="image-strip">
                    <img src={item.grid_url} alt={`${item.image} grid`} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </article>

        <article>
          <header>
            <h2>2. Edges & Corners</h2>
            <p>Canny-style edges plus Harris corner keypoints with interactive overlays.</p>
          </header>
          <form
            className="mini-form"
            onSubmit={(evt: FormEvent) => {
              evt.preventDefault();
              runFeatures();
            }}
          >
            <label>
              Sigma
              <input type="number" step={0.1} value={edgeSigma} onChange={(e) => setEdgeSigma(Number(e.target.value))} />
            </label>
            <label>
              Low / High thresholds
              <div className="inline-field">
                <input type="number" value={lowThresh} onChange={(e) => setLowThresh(Number(e.target.value))} />
                <input type="number" value={highThresh} onChange={(e) => setHighThresh(Number(e.target.value))} />
              </div>
            </label>
            <label>
              Corner threshold
              <input type="number" step={0.01} value={cornerThresh} onChange={(e) => setCornerThresh(Number(e.target.value))} />
            </label>
            <label>
              NMS radius
              <input type="number" min={2} value={nmsRadius} onChange={(e) => setNmsRadius(Number(e.target.value))} />
            </label>
            <label>
              Edge stride
              <input type="number" min={1} value={edgeStride} onChange={(e) => setEdgeStride(Number(e.target.value))} />
            </label>
            <button type="submit" className="primary-btn" disabled={featureLoading}>
              {featureLoading ? "Processing…" : "Run Edge/Corners"}
            </button>
          </form>
          {featureResults && (
            <div className="result-list">
              {featureResults.map((item) => (
                <div key={item.image} className="result-card">
                  <h3>{item.image}</h3>
                  <p>Edge pixels: {item.edge_pixel_count}</p>
                  <p>Edge keypoints: {item.edge_keypoint_count}</p>
                  <p>Corners: {item.corner_count}</p>
                  <div className="image-strip">
                    <img src={item.corner_overlay_url} alt={`${item.image} corners`} />
                    <img src={item.edges_overlay_url} alt={`${item.image} edges`} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </article>

        <article>
          <header>
            <h2>3. Object Boundaries</h2>
            <p>Adaptive Canny + contour scoring to select the best boundary per frame.</p>
          </header>
          <button type="button" className="primary-btn" onClick={runBoundaries} disabled={boundaryLoading}>
            {boundaryLoading ? "Processing…" : "Run Boundary Extraction"}
          </button>
          {boundaryResults && (
            <div className="result-list">
              {boundaryResults.map((item) => (
                <div key={item.image} className="result-card">
                  <h3>{item.image}</h3>
                  {item.found ? (
                    <>
                      <p>Score: {item.score?.toFixed(3)}</p>
                      <p>Rectangularity: {item.rectangularity?.toFixed(3)}</p>
                      <div className="image-strip">
                        <img src={item.bbox_overlay_url} alt={`${item.image} bbox`} />
                      </div>
                    </>
                  ) : (
                    <p>No valid contour detected.</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </article>
      </div>
    </section>
  );
}

export default ModuleThreePage;
