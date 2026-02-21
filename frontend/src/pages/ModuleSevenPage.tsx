import axios from "axios";
import { FormEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import ImageModal from "../components/ImageModal";
import { api, resolveApiUrl } from "../api/client";
import "./ModuleSevenPage.css";

type Point = { x: number; y: number };
type Point3D = { x: number; y: number; z: number };

type StereoMeasureResponse = {
  ok: boolean;
  session_id: string;
  points_3d: Point3D[];
  disparities: number[];
  edges_m: number[];
  edges_cm: number[];
  closed: boolean;
  summary: Record<string, number>;
  point_count: number;
  annotated_left_url?: string;
  annotated_right_url?: string;
};

type ShapeMode = "segment" | "circle" | "rectangle" | "polygon";

type Dimensions = { width: number; height: number };

type AnnotatedImage = { src: string; title: string } | null;

type PoseResponse = {
  ok: boolean;
  session_id: string;
  annotated_video_url: string;
  csv_url: string;
  processed_frames: number;
  input_frames: number;
  fps: number;
  width: number;
  height: number;
  duration_sec: number;
  sample_stride: number;
};

const DEFAULT_INTRINSICS = {
  fx: 4123.93,
  fy: 4124.5,
  cx: 1594.35,
  cy: 2864.54,
};

const DEFAULT_BASELINE = 0.0254;

const SHAPE_REQUIREMENTS: Record<ShapeMode, number | null> = {
  segment: 2,
  circle: 2,
  rectangle: 4,
  polygon: null, // at least 3
};

const formatBytes = (value: number) => {
  if (value >= 1024 * 1024) {
    const result = value / (1024 * 1024);
    return `${result.toFixed(2).replace(/\.00$/, "")} MB`;
  }
  if (value >= 1024) {
    const result = value / 1024;
    return `${result.toFixed(1).replace(/\.0$/, "")} KB`;
  }
  return `${value} B`;
};

const LIMITS = {
  stereoFileBytes: 4 * 1024 * 1024,
  stereoMaxPixels: 5_000_000,
  stereoMaxPoints: 40,
  poseFileBytes: 60 * 1024 * 1024,
  poseMaxFrames: 900,
  poseMaxStride: 30,
} as const;

const stereoFileLimitLabel = formatBytes(LIMITS.stereoFileBytes);
const poseFileLimitLabel = formatBytes(LIMITS.poseFileBytes);
const stereoMaxMpLabel = (LIMITS.stereoMaxPixels / 1_000_000).toFixed(1).replace(/\.0$/, "") + " MP";
const poseFrameLimitLabel = LIMITS.poseMaxFrames.toLocaleString();
const poseStrideLimitLabel = LIMITS.poseMaxStride.toLocaleString();
const clampPositive = (value: number, fallback: number) => (Number.isFinite(value) && value > 0 ? value : fallback);
const clampConfidence = (value: number) => Math.min(Math.max(Number.isFinite(value) ? value : 0.5, 0), 1);

const guardrailItems = [
  `Stereo uploads must stay under ${stereoFileLimitLabel} and ${stereoMaxMpLabel} per eye.`,
  `Limit ${LIMITS.stereoMaxPoints} correspondence points per image to keep disparity solves stable.`,
  `Pose clips must be under ${poseFileLimitLabel} with at most ${poseFrameLimitLabel} frames and stride ≤ ${poseStrideLimitLabel}.`,
  "Outputs are auto-pruned on the server—download MP4/CSV promptly if you need to keep them.",
] as const;

export function ModuleSevenPage() {
  const [leftFile, setLeftFile] = useState<File | null>(null);
  const [rightFile, setRightFile] = useState<File | null>(null);
  const [leftPreview, setLeftPreview] = useState<string | null>(null);
  const [rightPreview, setRightPreview] = useState<string | null>(null);
  const [leftPoints, setLeftPoints] = useState<Point[]>([]);
  const [rightPoints, setRightPoints] = useState<Point[]>([]);
  const [leftDims, setLeftDims] = useState<Dimensions | null>(null);
  const [rightDims, setRightDims] = useState<Dimensions | null>(null);
  const [shapeMode, setShapeMode] = useState<ShapeMode>("rectangle");
  const [fx, setFx] = useState(DEFAULT_INTRINSICS.fx);
  const [fy, setFy] = useState(DEFAULT_INTRINSICS.fy);
  const [cx, setCx] = useState(DEFAULT_INTRINSICS.cx);
  const [cy, setCy] = useState(DEFAULT_INTRINSICS.cy);
  const [baseline, setBaseline] = useState(DEFAULT_BASELINE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StereoMeasureResponse | null>(null);
  const [modalImage, setModalImage] = useState<AnnotatedImage>(null);
  const [poseFile, setPoseFile] = useState<File | null>(null);
  const [posePreview, setPosePreview] = useState<string | null>(null);
  const [poseStride, setPoseStride] = useState(1);
  const [poseMaxFrames, setPoseMaxFrames] = useState<number | "">("");
  const [poseDetection, setPoseDetection] = useState(0.5);
  const [poseTracking, setPoseTracking] = useState(0.5);
  const [poseLoading, setPoseLoading] = useState(false);
  const [poseError, setPoseError] = useState<string | null>(null);
  const [poseResult, setPoseResult] = useState<PoseResponse | null>(null);

  const leftImageRef = useRef<HTMLImageElement | null>(null);
  const rightImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => () => {
    if (leftPreview) URL.revokeObjectURL(leftPreview);
  }, [leftPreview]);

  useEffect(() => () => {
    if (rightPreview) URL.revokeObjectURL(rightPreview);
  }, [rightPreview]);

  useEffect(() => () => {
    if (posePreview) URL.revokeObjectURL(posePreview);
  }, [posePreview]);

  const expectedPoints = SHAPE_REQUIREMENTS[shapeMode];
  const closedPolygon = shapeMode === "rectangle" || shapeMode === "polygon";
  const minPoints = shapeMode === "polygon" ? 3 : expectedPoints ?? 2;

  const leftSummary = useMemo(() => `${leftPoints.length} point${leftPoints.length === 1 ? "" : "s"}`, [leftPoints]);
  const rightSummary = useMemo(() => `${rightPoints.length} point${rightPoints.length === 1 ? "" : "s"}`, [rightPoints]);

  const handleFileChange = (side: "left" | "right", file: File | null) => {
    if (file && file.size > LIMITS.stereoFileBytes) {
      setError(`${side === "left" ? "Left" : "Right"} image must be under ${stereoFileLimitLabel}.`);
      if (side === "left") {
        setLeftFile(null);
      } else {
        setRightFile(null);
      }
      return;
    }
    const assign = side === "left" ? setLeftFile : setRightFile;
    const setPreview = side === "left" ? setLeftPreview : setRightPreview;
    const setDims = side === "left" ? setLeftDims : setRightDims;
    assign(file);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setDims(null);
    if (side === "left") setLeftPoints([]);
    else setRightPoints([]);
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
    setResult(null);
    setError(null);
  };

  const handlePoseFileChange = (file: File | null) => {
    if (file && file.size > LIMITS.poseFileBytes) {
      setPoseError(`Video upload must be under ${poseFileLimitLabel}.`);
      setPoseFile(null);
      return;
    }
    setPoseFile(file);
    setPosePreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (file) {
      setPosePreview(URL.createObjectURL(file));
    }
    setPoseResult(null);
    setPoseError(null);
  };

  const extractPoint = (event: MouseEvent<HTMLDivElement>, side: "left" | "right") => {
    const imageRef = side === "left" ? leftImageRef : rightImageRef;
    const dims = side === "left" ? leftDims : rightDims;
    if (!imageRef.current || !dims) return null;
    const rect = imageRef.current.getBoundingClientRect();
    const relX = (event.clientX - rect.left) / rect.width;
    const relY = (event.clientY - rect.top) / rect.height;
    if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return null;
    return {
      x: relX * dims.width,
      y: relY * dims.height,
    };
  };

  const handlePreviewClick = (event: MouseEvent<HTMLDivElement>, side: "left" | "right") => {
    const point = extractPoint(event, side);
    if (!point) return;
    const limit = LIMITS.stereoMaxPoints;
    if (limit > 0) {
      const currentCount = side === "left" ? leftPoints.length : rightPoints.length;
      if (currentCount >= limit) {
        setError(`Stereo correspondences are capped at ${LIMITS.stereoMaxPoints} points per image.`);
        return;
      }
    }
    if (side === "left") {
      setLeftPoints((prev) => [...prev, point]);
    } else {
      setRightPoints((prev) => [...prev, point]);
    }
    setError(null);
    setResult(null);
  };

  const undoPoint = (side: "left" | "right") => {
    if (side === "left") {
      setLeftPoints((prev) => prev.slice(0, -1));
    } else {
      setRightPoints((prev) => prev.slice(0, -1));
    }
  };

  const resetPoints = () => {
    setLeftPoints([]);
    setRightPoints([]);
    setResult(null);
  };

  const getPointStyle = (pt: Point, dims: Dimensions | null) => {
    if (!dims) return {};
    return {
      left: `${(pt.x / dims.width) * 100}%`,
      top: `${(pt.y / dims.height) * 100}%`,
    };
  };

  const canSubmit =
    leftFile &&
    rightFile &&
    leftPoints.length === rightPoints.length &&
    leftPoints.length >= minPoints &&
    (expectedPoints === null || leftPoints.length === expectedPoints);

  const buildFormPoints = (points: Point[]) => points.map((pt) => ({ x: pt.x, y: pt.y }));

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit || !leftFile || !rightFile) {
      setError("Upload both images and add matching points first.");
      return;
    }
    const safeFx = clampPositive(fx, DEFAULT_INTRINSICS.fx);
    const safeFy = clampPositive(fy, DEFAULT_INTRINSICS.fy);
    const safeCx = clampPositive(cx, DEFAULT_INTRINSICS.cx);
    const safeCy = clampPositive(cy, DEFAULT_INTRINSICS.cy);
    const safeBaseline = clampPositive(baseline, DEFAULT_BASELINE);

    const form = new FormData();
    form.append("left_image", leftFile);
    form.append("right_image", rightFile);
    form.append("fx", String(safeFx));
    form.append("fy", String(safeFy));
    form.append("cx", String(safeCx));
    form.append("cy", String(safeCy));
    form.append("baseline", String(safeBaseline));
    form.append("shape", shapeMode);
    form.append("closed", String(closedPolygon));
    form.append("points_left", JSON.stringify(buildFormPoints(leftPoints)));
    form.append("points_right", JSON.stringify(buildFormPoints(rightPoints)));
    try {
      setLoading(true);
      setError(null);
      const { data } = await api.post<StereoMeasureResponse>("/api/module7/stereo-measure", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const annotatedLeft = data.annotated_left_url ? resolveApiUrl(data.annotated_left_url) : undefined;
      const annotatedRight = data.annotated_right_url ? resolveApiUrl(data.annotated_right_url) : undefined;
      setResult({ ...data, annotated_left_url: annotatedLeft, annotated_right_url: annotatedRight });
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err)
        ? (err.response?.data as { detail?: string } | undefined)?.detail
        : undefined;
      setError(detail ?? (err instanceof Error ? err.message : "Failed to run stereo measurement."));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePoseSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!poseFile) {
      setPoseError("Upload a video clip first.");
      return;
    }
    const safeStride = Math.min(Math.max(1, poseStride), LIMITS.poseMaxStride);
    const safeMaxFrames =
      poseMaxFrames === "" ? undefined : Math.min(Math.max(1, Number(poseMaxFrames)), LIMITS.poseMaxFrames);
    const safeDetection = clampConfidence(poseDetection);
    const safeTracking = clampConfidence(poseTracking);

    const form = new FormData();
    form.append("video", poseFile);
    form.append("sample_stride", String(safeStride));
    if (safeMaxFrames !== undefined) {
      form.append("max_frames", String(safeMaxFrames));
    }
    form.append("detection_confidence", String(safeDetection));
    form.append("tracking_confidence", String(safeTracking));

    try {
      setPoseLoading(true);
      setPoseError(null);
      const { data } = await api.post<PoseResponse>("/api/module7/pose", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPoseResult({
        ...data,
        annotated_video_url: resolveApiUrl(data.annotated_video_url),
        csv_url: resolveApiUrl(data.csv_url),
      });
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err)
        ? (err.response?.data as { detail?: string } | undefined)?.detail
        : undefined;
      setPoseError(detail ?? (err instanceof Error ? err.message : "Failed to run pose tracking."));
      setPoseResult(null);
    } finally {
      setPoseLoading(false);
    }
  };

  const renderPointList = (points: Point[]) => (
    <ol className="point-list">
      {points.map((pt, idx) => (
        <li key={`${pt.x}-${pt.y}-${idx}`}>
          P{idx}: ({pt.x.toFixed(1)}, {pt.y.toFixed(1)})
        </li>
      ))}
    </ol>
  );

  const renderSummary = () => {
    if (!result) return null;
    return (
      <div className="stereo-summary">
        {Object.entries(result.summary || {}).map(([key, value]) => (
          <div key={key} className="summary-chip">
            <p className="chip-label">{key.replace(/_/g, " ")}</p>
            <p className="chip-value">{value.toFixed(4)}</p>
          </div>
        ))}
      </div>
    );
  };

  const renderEdges = () => {
    if (!result?.edges_cm?.length) return null;
    return (
      <div>
        <h3>Edge Lengths</h3>
        <ul className="edge-list">
          {result.edges_cm.map((value, idx) => (
            <li key={`edge-${idx}`}>
              Edge {idx + 1}: {value.toFixed(2)} cm ({result.edges_m[idx]?.toFixed(4)} m)
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderPoints3d = () => {
    if (!result?.points_3d?.length) return null;
    return (
      <div>
        <h3>Triangulated Points</h3>
        <table className="points-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>X (m)</th>
              <th>Y (m)</th>
              <th>Z (m)</th>
              <th>Disparity (px)</th>
            </tr>
          </thead>
          <tbody>
            {result.points_3d.map((pt, idx) => (
              <tr key={`pt-${idx}`}>
                <td>P{idx}</td>
                <td>{pt.x.toFixed(4)}</td>
                <td>{pt.y.toFixed(4)}</td>
                <td>{pt.z.toFixed(4)}</td>
                <td>{result.disparities?.[idx]?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderAnnotated = () => {
    if (!result?.annotated_left_url && !result?.annotated_right_url) return null;
    return (
      <div className="annotated-grid">
        {result.annotated_left_url && (
          <figure>
            <img
              src={result.annotated_left_url}
              alt="Left annotated"
              onClick={() => setModalImage({ src: result.annotated_left_url!, title: "Left annotated" })}
            />
            <figcaption>Left view</figcaption>
          </figure>
        )}
        {result.annotated_right_url && (
          <figure>
            <img
              src={result.annotated_right_url}
              alt="Right annotated"
              onClick={() => setModalImage({ src: result.annotated_right_url!, title: "Right annotated" })}
            />
            <figcaption>Right view</figcaption>
          </figure>
        )}
      </div>
    );
  };

  const renderPoseMetadata = () => {
    if (!poseResult) return null;
    const metrics = [
      { label: "Processed frames", value: poseResult.processed_frames },
      { label: "Input frames", value: poseResult.input_frames },
      { label: "FPS", value: poseResult.fps.toFixed(1) },
      { label: "Duration (s)", value: poseResult.duration_sec.toFixed(2) },
      { label: "Resolution", value: `${poseResult.width}×${poseResult.height}` },
      { label: "Stride", value: poseResult.sample_stride },
    ];
    return (
      <div className="pose-metrics">
        {metrics.map((metric) => (
          <div key={metric.label} className="summary-chip">
            <p className="chip-label">{metric.label}</p>
            <p className="chip-value">{metric.value}</p>
          </div>
        ))}
      </div>
    );
  };

  return (
    <section className="module7-root">
      <header className="module7-header">
        <p className="eyebrow">Calibrated Stereo Lab</p>
        <h1>Stereo Object Size Estimation</h1>
        <p>
          Upload rectified stereo pairs, place matching correspondences, and convert disparities into real-world dimensions using your
          calibrated rig. Perfect for estimating rectangular footprints, polygon edges, or circular diameters in 3D.
        </p>
      </header>

      <div className="guardrail-banner" aria-live="polite">
        <p className="guardrail-title">Render guardrails in effect</p>
        <ul className="guardrail-list">
          {guardrailItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <form className="module7-grid" onSubmit={handleSubmit}>
        <article className="stereo-panel">
          <h2>Stereo Inputs</h2>
          <label className="field">
            <span>Left image</span>
            <input type="file" accept="image/*" onChange={(event) => handleFileChange("left", event.target.files?.[0] ?? null)} />
            <small>{leftFile ? leftFile.name : "No file selected"}</small>
            <p className="field-hint guardrail-hint">
              Max {stereoFileLimitLabel} per file, ≤ {stereoMaxMpLabel}, {LIMITS.stereoMaxPoints} matches per eye.
            </p>
          </label>
          <label className="field">
            <span>Right image</span>
            <input type="file" accept="image/*" onChange={(event) => handleFileChange("right", event.target.files?.[0] ?? null)} />
            <small>{rightFile ? rightFile.name : "No file selected"}</small>
          </label>

          <label className="field">
            <span>Shape mode</span>
            <select value={shapeMode} onChange={(event) => setShapeMode(event.target.value as ShapeMode)}>
              <option value="rectangle">Rectangle</option>
              <option value="segment">Line / segment</option>
              <option value="circle">Circle diameter</option>
              <option value="polygon">Polygon</option>
            </select>
            <small>
              {shapeMode === "rectangle" && "Provide 4 ordered corners (closed polygon)."}
              {shapeMode === "segment" && "Provide exactly 2 matching points."}
              {shapeMode === "circle" && "Provide 2 points across the diameter."}
              {shapeMode === "polygon" && "Provide 3+ points; we'll treat it as closed."}
            </small>
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
          <div className="field-grid">
            <label className="field">
              <span>cx</span>
              <input type="number" step={0.01} value={cx} onChange={(event) => setCx(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>cy</span>
              <input type="number" step={0.01} value={cy} onChange={(event) => setCy(Number(event.target.value))} />
            </label>
          </div>
          <label className="field">
            <span>Baseline B (meters)</span>
            <input type="number" step={0.0001} value={baseline} onChange={(event) => setBaseline(Number(event.target.value))} />
          </label>

          <div className="field-row spaced">
            <p className="field-hint">Left: {leftSummary} · Right: {rightSummary}</p>
            <div className="button-cluster">
              <button type="button" className="ghost-btn" onClick={() => undoPoint("left")} disabled={!leftPoints.length}>
                Undo left
              </button>
              <button type="button" className="ghost-btn" onClick={() => undoPoint("right")} disabled={!rightPoints.length}>
                Undo right
              </button>
              <button type="button" className="ghost-btn" onClick={resetPoints} disabled={!leftPoints.length && !rightPoints.length}>
                Reset points
              </button>
            </div>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="primary-btn" disabled={!canSubmit || loading}>
            {loading ? "Computing…" : "Triangulate dimensions"}
          </button>
        </article>

        <article className="stereo-preview-panel">
          <h2>Left View</h2>
          <div className="stereo-preview" onClick={(event) => handlePreviewClick(event, "left")}>
            {leftPreview ? (
              <>
                <img
                  ref={leftImageRef}
                  src={leftPreview}
                  alt="Left preview"
                  onLoad={(event) =>
                    setLeftDims({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })
                  }
                />
                {leftDims && (
                  <div className="stereo-overlay">
                    {leftPoints.map((pt, idx) => (
                      <span key={`left-${pt.x}-${pt.y}-${idx}`} style={getPointStyle(pt, leftDims)}>
                        {idx}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="placeholder">Upload a left image to begin marking correspondences.</p>
            )}
          </div>
          {renderPointList(leftPoints)}
        </article>

        <article className="stereo-preview-panel">
          <h2>Right View</h2>
          <div className="stereo-preview" onClick={(event) => handlePreviewClick(event, "right")}>
            {rightPreview ? (
              <>
                <img
                  ref={rightImageRef}
                  src={rightPreview}
                  alt="Right preview"
                  onLoad={(event) =>
                    setRightDims({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })
                  }
                />
                {rightDims && (
                  <div className="stereo-overlay">
                    {rightPoints.map((pt, idx) => (
                      <span key={`right-${pt.x}-${pt.y}-${idx}`} style={getPointStyle(pt, rightDims)}>
                        {idx}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="placeholder">Upload the matching right image and click the same points.</p>
            )}
          </div>
          {renderPointList(rightPoints)}
        </article>
      </form>

      <section className="stereo-results">
        <h2>Results</h2>
        {!result && <p className="placeholder">Submit a stereo pair with matching correspondences to see the 3D estimates.</p>}
        {result && (
          <div className="results-grid">
            {renderSummary()}
            {renderEdges()}
            {renderPoints3d()}
            {renderAnnotated()}
          </div>
        )}
      </section>

      <section className="pose-root">
        <header className="pose-header">
          <p className="eyebrow">Pose + Hands</p>
          <h2>Real-time Pose Estimation</h2>
          <p>
            Drop a short webcam capture or studio clip, then let MediaPipe’s holistic tracker recover full-body joints and both hands. We’ll
            return an annotated MP4 plus a CSV log so you can document every landmark position in your report.
          </p>
        </header>

        <div className="pose-grid">
          <form className="pose-panel" onSubmit={handlePoseSubmit}>
            <h3>Upload & Parameters</h3>
            <label className="field">
              <span>Video capture</span>
              <input type="file" accept="video/*" onChange={(event) => handlePoseFileChange(event.target.files?.[0] ?? null)} />
              <small>{poseFile ? poseFile.name : "No file selected"}</small>
              <p className="field-hint guardrail-hint">Keep clips under {poseFileLimitLabel}; longer clips are rejected server-side.</p>
            </label>

            <div className="field-grid">
              <label className="field">
                <span>Sample stride</span>
                <input
                  type="number"
                  min={1}
                  max={LIMITS.poseMaxStride}
                  value={poseStride}
                  onChange={(event) => setPoseStride(Math.max(1, Number(event.target.value)))}
                />
                <small>
                  Process every Nth frame (default 1, max {poseStrideLimitLabel}). Larger entries are auto-clamped.
                </small>
              </label>
              <label className="field">
                <span>Max frames (optional)</span>
                <input
                  type="number"
                  min={1}
                  max={LIMITS.poseMaxFrames}
                  value={poseMaxFrames}
                  onChange={(event) =>
                    setPoseMaxFrames(event.target.value === "" ? "" : Math.max(1, Number(event.target.value)))
                  }
                />
                <small>Leave blank to process full clip (cap {poseFrameLimitLabel} frames per run).</small>
              </label>
            </div>

            <div className="field-grid">
              <label className="field">
                <span>Detection confidence</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={poseDetection}
                  onChange={(event) => setPoseDetection(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Tracking confidence</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={poseTracking}
                  onChange={(event) => setPoseTracking(Number(event.target.value))}
                />
              </label>
            </div>

            {poseError && <p className="form-error">{poseError}</p>}

            <button type="submit" className="primary-btn" disabled={!poseFile || poseLoading}>
              {poseLoading ? "Running pose tracker…" : "Run pose tracking"}
            </button>
          </form>

          <article className="pose-results-panel">
            <h3>Outputs</h3>
            <div className="pose-preview">
              {poseResult?.annotated_video_url ? (
                <video src={poseResult.annotated_video_url} controls autoPlay loop muted />
              ) : posePreview ? (
                <video src={posePreview} controls loop muted />
              ) : (
                <p className="placeholder">Upload a clip to preview and collect pose data.</p>
              )}
            </div>

            {poseResult?.csv_url && (
              <a className="download-link" href={poseResult.csv_url} target="_blank" rel="noreferrer">
                Download CSV log
              </a>
            )}

            {renderPoseMetadata()}
          </article>
        </div>
      </section>

      <ImageModal open={Boolean(modalImage)} onClose={() => setModalImage(null)} title={modalImage?.title}>
        {modalImage && <img src={modalImage.src} alt={modalImage.title} />}
      </ImageModal>
    </section>
  );
}

export default ModuleSevenPage;
