import { FormEvent, useEffect, useMemo, useState } from "react";
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

type SegmentationResult = {
  image: string;
  marker_count: number;
  hull_area_px?: number;
  perimeter_px?: number;
  coverage_pct?: number;
  mask_url?: string;
  overlay_url?: string;
  contour_url?: string;
  markers_debug_url?: string;
  sam_mask_url?: string;
  sam_overlay_url?: string;
  sam_quality?: number;
  sam_iou?: number;
  sam_dice?: number;
  sam_error?: string;
  issues?: string;
};

type SegmentationSummary = {
  total_images: number;
  success_count: number;
  avg_marker_count?: number;
  avg_area_px?: number;
  avg_sam_iou?: number;
  avg_sam_dice?: number;
  sam_model_id?: string;
  summary_csv_url?: string;
  uploaded_mask_matches?: number;
};

type DatasetEntry = {
  name: string;
  size: number;
  modified: number;
};

type DatasetListResponse = {
  ok: boolean;
  entries: DatasetEntry[];
  count: number;
  sam2_inference_enabled?: boolean;
};

type DatasetUploadResponse = {
  ok: boolean;
  saved: { original: string; stored_as: string; size: number }[];
  total_images: number;
};

type Module3LibraryChunkResponse<T> = {
  ok: boolean;
  results: T[];
  batch: string;
  source: "library";
  cursor: number;
  next_cursor: number | null;
  total_images: number;
  processed_count: number;
  chunk_size: number;
};

type LibraryQuality = "fast" | "balanced" | "high" | "full";

const LIBRARY_QUALITY_MAX_DIM: Record<LibraryQuality, number | null> = {
  fast: 256,
  balanced: 512,
  high: 768,
  full: null,
};

const LIBRARY_QUALITY_OPTIONS: LibraryQuality[] = ["fast", "balanced", "high", "full"];

const LIBRARY_QUALITY_DESCRIPTIONS: Record<LibraryQuality, string> = {
  fast: "Fastest (≤256 px longest edge).",
  balanced: "Balanced (≤512 px) – solid quality on Render.",
  high: "High fidelity (≤768 px) – better detail, slightly slower.",
  full: "Full resolution (no downscaling). Might timeout on Render.",
};

const DATASET_MAX_IMAGES = 60;
const DATASET_MAX_SIZE_KB = 2048;
const EDGE_UPLOAD_LIMIT = 25;
const EDGE_UPLOAD_MAX_KB = 4096;
const SEG_UPLOAD_LIMIT = 30;
const SEG_UPLOAD_MAX_KB = 5120;

const formatSizeLimit = (kb: number) => (kb % 1024 === 0 ? `${kb / 1024} MB` : `${kb} KB`);

const toAbsolute = (path: string) => {
  if (!path) return path;
  try {
    return path.startsWith("http") ? path : new URL(path, api.defaults.baseURL).toString();
  } catch {
    return path;
  }
};

const fetchModule3LibraryChunks = async <T,>(
  path: string,
  populate: (formData: FormData) => void,
): Promise<T[]> => {
  const merged: T[] = [];
  let cursor = 0;
  const loopGuard = 512;
  for (let step = 0; step < loopGuard; step += 1) {
    const formData = new FormData();
    populate(formData);
    formData.set("cursor", String(cursor));
    const { data } = await api.post<Module3LibraryChunkResponse<T>>(path, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    if (Array.isArray(data.results)) {
      merged.push(...data.results);
    }
    if (data.next_cursor == null) {
      break;
    }
    if (data.next_cursor === cursor) {
      console.warn(`[Module3] Pagination stalled at cursor ${cursor}.`);
      break;
    }
    cursor = data.next_cursor;
  }
  return merged;
};

export function ModuleThreePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [segFiles, setSegFiles] = useState<File[]>([]);
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
  const [segResults, setSegResults] = useState<SegmentationResult[] | null>(null);
  const [segSummary, setSegSummary] = useState<SegmentationSummary | null>(null);

  const [gradLoading, setGradLoading] = useState(false);
  const [featureLoading, setFeatureLoading] = useState(false);
  const [boundaryLoading, setBoundaryLoading] = useState(false);
  const [segLoading, setSegLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [datasetEntries, setDatasetEntries] = useState<DatasetEntry[]>([]);
  const [datasetListLoading, setDatasetListLoading] = useState(false);
  const [datasetUploadFiles, setDatasetUploadFiles] = useState<File[]>([]);
  const [datasetUploadLoading, setDatasetUploadLoading] = useState(false);
  const [datasetResetLoading, setDatasetResetLoading] = useState(false);
  const [datasetManagerMessage, setDatasetManagerMessage] = useState<string | null>(null);
  const [datasetManagerError, setDatasetManagerError] = useState<string | null>(null);
  const [datasetSource, setDatasetSource] = useState<"upload" | "library">("upload");
  const [libraryQuality, setLibraryQuality] = useState<LibraryQuality>("balanced");

  const [dictionary, setDictionary] = useState("DICT_4X4_100");
  const [minMarkers, setMinMarkers] = useState(4);
  const [dilatePx, setDilatePx] = useState(25);
  const [smoothPx, setSmoothPx] = useState(11);
  const [compareSam, setCompareSam] = useState(true);
  const [samModelId, setSamModelId] = useState("facebook/sam2.1-hiera-tiny");
  const [samMaskBundle, setSamMaskBundle] = useState<File | null>(null);
  const [sam2BackendEnabled, setSam2BackendEnabled] = useState(false);

  const fetchDatasetEntries = async () => {
    setDatasetListLoading(true);
    try {
      const { data } = await api.get<DatasetListResponse>("/api/module3/dataset");
      setDatasetEntries(data.entries ?? []);
      setSam2BackendEnabled(Boolean(data.sam2_inference_enabled));
    } catch (err) {
      console.error(err);
      setDatasetManagerError(err instanceof Error ? err.message : "Unable to load dataset library.");
    } finally {
      setDatasetListLoading(false);
    }
  };

  useEffect(() => {
    void fetchDatasetEntries();
  }, []);

  useEffect(() => {
    if (!sam2BackendEnabled && compareSam) {
      setCompareSam(false);
    }
  }, [sam2BackendEnabled, compareSam]);

  const datasetUploadSummary = useMemo(() => {
    if (datasetUploadFiles.length === 0) {
      return `Select ≥10 aligned frames captured from different views (≤${formatSizeLimit(DATASET_MAX_SIZE_KB)} each).`;
    }
    if (datasetUploadFiles.length === 1) {
      return `${datasetUploadFiles[0].name} · library stores up to ${DATASET_MAX_IMAGES} images (≤${formatSizeLimit(DATASET_MAX_SIZE_KB)} each).`;
    }
    return `${datasetUploadFiles.length} files ready (${datasetUploadFiles.map((file) => file.name).join(", ")}) · max ${DATASET_MAX_IMAGES} images stored.`;
  }, [datasetUploadFiles]);

  const datasetLibrarySummary = useMemo(() => {
    if (datasetEntries.length === 0) {
      return "No dataset stored yet.";
    }
    if (datasetEntries.length === 1) {
      return datasetEntries[0].name;
    }
    return `${datasetEntries.length} stored images ready.`;
  }, [datasetEntries]);

  const libraryQualitySummary = useMemo(() => LIBRARY_QUALITY_DESCRIPTIONS[libraryQuality], [libraryQuality]);

  const applyLibraryMaxDim = (formData: FormData) => {
    const limit = LIBRARY_QUALITY_MAX_DIM[libraryQuality];
    if (limit == null) {
      formData.set("max_dim", "0");
      return;
    }
    formData.set("max_dim", String(limit));
  };

  const hasStoredDataset = datasetEntries.length > 0;

  const submitDatasetUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (datasetUploadFiles.length === 0) {
      setDatasetManagerError("Select at least one dataset image.");
      return;
    }

    const formData = new FormData();
    datasetUploadFiles.forEach((file) => formData.append("images", file));

    try {
      setDatasetUploadLoading(true);
      setDatasetManagerError(null);
      const { data } = await api.post<DatasetUploadResponse>("/api/module3/dataset_upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDatasetManagerMessage(
        `Uploaded ${data.saved.length} image${data.saved.length === 1 ? "" : "s"}. Total stored: ${data.total_images}.`,
      );
      setDatasetUploadFiles([]);
      form.reset();
      await fetchDatasetEntries();
    } catch (err) {
      setDatasetManagerError(err instanceof Error ? err.message : "Dataset upload failed.");
      setDatasetManagerMessage(null);
    } finally {
      setDatasetUploadLoading(false);
    }
  };

  const handleDatasetReset = async () => {
    if (!window.confirm("Delete all stored dataset images?")) {
      return;
    }
    try {
      setDatasetResetLoading(true);
      setDatasetManagerError(null);
      setDatasetManagerMessage(null);
      await api.delete("/api/module3/dataset");
      await fetchDatasetEntries();
      setDatasetSource("upload");
      setDatasetManagerMessage("Dataset library cleared.");
    } catch (err) {
      setDatasetManagerError(err instanceof Error ? err.message : "Unable to reset dataset library.");
      setDatasetManagerMessage(null);
    } finally {
      setDatasetResetLoading(false);
    }
  };

  const fileSummary = useMemo(() => {
    if (datasetSource === "library") {
      if (!datasetEntries.length) {
        return "No dataset stored yet.";
      }
      return `${datasetEntries.length} stored image${datasetEntries.length > 1 ? "s" : ""} ready.`;
    }
    const uploadLimitCopy = `Limit ${EDGE_UPLOAD_LIMIT} images/run (≤${formatSizeLimit(EDGE_UPLOAD_MAX_KB)} each).`;
    if (!files.length) return `No images selected yet. ${uploadLimitCopy}`;
    return `${files.length} image${files.length > 1 ? "s" : ""} selected. ${uploadLimitCopy}`;
  }, [datasetEntries, datasetSource, files]);

  const segFileSummary = useMemo(() => {
    if (!segFiles.length) {
      return `No segmentation dataset selected. Limit ${SEG_UPLOAD_LIMIT} images/run (≤${formatSizeLimit(SEG_UPLOAD_MAX_KB)} each).`;
    }
    return `${segFiles.length} image${segFiles.length > 1 ? "s" : ""} ready for ArUco/SAM2 (limit ${SEG_UPLOAD_LIMIT}).`;
  }, [segFiles]);

  const samMaskSummary = useMemo(() => {
    if (!samMaskBundle) {
      return "Optional: upload a .zip or .npz bundle of SAM masks (filenames should match your frames).";
    }
    return `${samMaskBundle.name} · ${(samMaskBundle.size / 1024).toFixed(1)} KB`;
  }, [samMaskBundle]);

  const sam2StatusMessage = sam2BackendEnabled
    ? "SAM2 inference is enabled for this backend."
    : "Live SAM2 inference was removed on this deployment. Upload SAM mask bundles generated offline if you need IoU/Dice metrics.";

  const segmentationButtonLabel = sam2BackendEnabled && compareSam ? "Run ArUco + SAM2" : "Run ArUco";

  const handleFiles = (list: FileList | null) => {
    if (!list) {
      setFiles([]);
      return;
    }
    const imgs = Array.from(list).filter((file) => file.type.startsWith("image/"));
    setFiles(imgs);
  };

  const handleSegFiles = (list: FileList | null) => {
    if (!list) {
      setSegFiles([]);
      return;
    }
    const imgs = Array.from(list).filter((file) => file.type.startsWith("image/"));
    setSegFiles(imgs);
  };

  const requireDataset = () => {
    if (datasetSource === "library") {
      if (!hasStoredDataset) {
        setError("Dataset library is empty. Upload images or switch to on-the-fly uploads.");
        return false;
      }
      setError(null);
      return true;
    }
    if (!files.length) {
      setError("Please upload at least one object image.");
      return false;
    }
    setError(null);
    return true;
  };

  const requireSegFiles = () => {
    if (!segFiles.length) {
      setError("Please upload your segmentation dataset (≥10 images).");
      return false;
    }
    setError(null);
    return true;
  };

  const runGradients = async () => {
    if (!requireDataset()) return;
    if (datasetSource === "library") {
      try {
        setGradLoading(true);
        setError(null);
        const results = await fetchModule3LibraryChunks<GradientResult>(
          "/api/module3/library_gradients",
          (formData) => {
            formData.append("sigma", String(sigma));
            formData.append("ksize", String(logKsize));
            applyLibraryMaxDim(formData);
          },
        );
        setGradResults(
          results.map((item) => ({
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
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));
    formData.append("sigma", String(sigma));
    formData.append("ksize", String(logKsize));
    try {
      setGradLoading(true);
      setError(null);
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
    if (!requireDataset()) return;
    if (datasetSource === "library") {
      try {
        setFeatureLoading(true);
        setError(null);
        const results = await fetchModule3LibraryChunks<FeatureResult>(
          "/api/module3/library_features",
          (formData) => {
            formData.append("sigma", String(edgeSigma));
            formData.append("low", String(lowThresh));
            formData.append("high", String(highThresh));
            formData.append("corner_thresh", String(cornerThresh));
            formData.append("nms_radius", String(nmsRadius));
            formData.append("edge_stride", String(edgeStride));
            applyLibraryMaxDim(formData);
          },
        );
        setFeatureResults(
          results.map((item) => ({
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
      return;
    }

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
      setError(null);
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
    if (!requireDataset()) return;
    if (datasetSource === "library") {
      try {
        setBoundaryLoading(true);
        setError(null);
        const results = await fetchModule3LibraryChunks<BoundaryResult>(
          "/api/module3/library_boundaries",
          (formData) => {
            formData.append("sigma", String(edgeSigma));
            formData.append("low", String(lowThresh));
            formData.append("high", String(highThresh));
            applyLibraryMaxDim(formData);
          },
        );
        setBoundaryResults(
          results.map((item) => ({
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
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));
    try {
      setBoundaryLoading(true);
      setError(null);
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

  const runSegmentation = async () => {
    if (!requireSegFiles()) return;
    const formData = new FormData();
    segFiles.forEach((file) => formData.append("images", file));
    formData.append("dictionary", dictionary);
    formData.append("min_markers", String(minMarkers));
    formData.append("dilate_px", String(dilatePx));
    formData.append("smooth_px", String(smoothPx));
    const requestSam = sam2BackendEnabled && compareSam;
    formData.append("compare_sam2", String(requestSam));
    if (samMaskBundle) {
      formData.append("sam2_masks", samMaskBundle);
    }
    if (samModelId.trim()) {
      formData.append("sam2_model_id", samModelId.trim());
    }
    try {
      setSegLoading(true);
      const { data } = await api.post<{ results: SegmentationResult[]; summary: SegmentationSummary }>("/api/module3/aruco_segment", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSegResults(
        data.results.map((item) => ({
          ...item,
          mask_url: item.mask_url ? toAbsolute(item.mask_url) : undefined,
          overlay_url: item.overlay_url ? toAbsolute(item.overlay_url) : undefined,
          contour_url: item.contour_url ? toAbsolute(item.contour_url) : undefined,
          markers_debug_url: item.markers_debug_url ? toAbsolute(item.markers_debug_url) : undefined,
          sam_mask_url: item.sam_mask_url ? toAbsolute(item.sam_mask_url) : undefined,
          sam_overlay_url: item.sam_overlay_url ? toAbsolute(item.sam_overlay_url) : undefined,
        }))
      );
      const summaryPayload: SegmentationSummary = {
        ...data.summary,
        summary_csv_url: data.summary.summary_csv_url ? toAbsolute(data.summary.summary_csv_url) : undefined,
      };
      setSegSummary(summaryPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run segmentation.");
    } finally {
      setSegLoading(false);
    }
  };

  return (
    <section className="module3-root">
      <header className="module3-header">
        <p className="eyebrow">Dataset Lab</p>
        <h1>Edge & Boundary Analysis</h1>
        <p>
          Upload a small dataset (≈10 images of your measurement object). Run gradient + LoG diagnostics, inspect adaptive edge/corner
          detections, and extract final object boundaries in one workflow.
        </p>
      </header>

      <article className="module3-card dataset-manager-card">
        <header>
          <h2>Dataset Library Manager</h2>
          <p>Capture your dataset once, store it on the server, and reuse it across all experiments.</p>
        </header>
        <form className="dataset-form" onSubmit={submitDatasetUpload}>
          <label className="field">
            <span>Library dataset files</span>
            <input
              type="file"
              accept="image/png, image/jpeg, image/webp"
              multiple
              onChange={(event) => setDatasetUploadFiles(Array.from(event.target.files ?? []))}
            />
            <small>{datasetUploadSummary}</small>
          </label>
          {datasetManagerError && <p className="form-error">{datasetManagerError}</p>}
          {datasetManagerMessage && <p className="form-note">{datasetManagerMessage}</p>}
          <button type="submit" className="secondary-btn" disabled={datasetUploadLoading}>
            {datasetUploadLoading ? "Uploading…" : "Upload to library"}
          </button>
        </form>
        <div className="dataset-actions">
          <button type="button" className="ghost-btn" onClick={() => void fetchDatasetEntries()} disabled={datasetListLoading}>
            {datasetListLoading ? "Refreshing…" : "Refresh list"}
          </button>
          <button
            type="button"
            className="danger-btn"
            onClick={handleDatasetReset}
            disabled={datasetResetLoading || !hasStoredDataset}
          >
            {datasetResetLoading ? "Clearing…" : "Reset library"}
          </button>
        </div>
        <p className="helper-text">Stored dataset: {datasetEntries.length} image(s). Recommended ≥10 unique captures.</p>
        <p className="helper-text">
          Server keeps up to {DATASET_MAX_IMAGES} images (≤{formatSizeLimit(DATASET_MAX_SIZE_KB)} each). Reset the library if you need more space.
        </p>
        {hasStoredDataset ? (
          <div className="dataset-list">
            {datasetEntries.slice(0, 8).map((entry) => (
              <div key={entry.name} className="dataset-row">
                <div className="dataset-name">{entry.name}</div>
                <div className="dataset-meta">
                  {(entry.size / 1024).toFixed(1)} KB · {new Date(entry.modified * 1000).toLocaleDateString()}
                </div>
              </div>
            ))}
            {datasetEntries.length > 8 && (
              <p className="helper-text">{datasetEntries.length - 8}+ additional images stored.</p>
            )}
          </div>
        ) : (
          <p className="result-placeholder">No dataset stored yet. Upload the frames you will reference in questions 1–3.</p>
        )}
      </article>

      <div className="upload-panel">
        <div className="dataset-source-switch">
          <label className={`radio-card ${datasetSource === "upload" ? "active" : ""}`}>
            <input
              type="radio"
              name="dataset-source"
              value="upload"
              checked={datasetSource === "upload"}
              onChange={() => setDatasetSource("upload")}
            />
            <div>
              <strong>Use ad-hoc uploads</strong>
              <p>Only uses the files you select below.</p>
            </div>
          </label>
          <label className={`radio-card ${datasetSource === "library" ? "active" : ""} ${!hasStoredDataset ? "disabled" : ""}`}>
            <input
              type="radio"
              name="dataset-source"
              value="library"
              checked={datasetSource === "library"}
              onChange={() => setDatasetSource("library")}
              disabled={!hasStoredDataset}
            />
            <div>
              <strong>Use stored dataset</strong>
              <p>{datasetLibrarySummary}</p>
            </div>
          </label>
        </div>
        <label className="field">
          <span>Dataset images</span>
          <input
            type="file"
            accept="image/*"
            multiple
            disabled={datasetSource === "library"}
            onChange={(event) => handleFiles(event.target.files)}
          />
          <small>{fileSummary}</small>
        </label>
        {datasetSource === "library" && (
          <label className="field">
            <span>Library image quality</span>
            <select value={libraryQuality} onChange={(event) => setLibraryQuality(event.target.value as LibraryQuality)}>
              {LIBRARY_QUALITY_OPTIONS.map((quality) => (
                <option key={quality} value={quality}>
                  {quality === "fast"
                    ? "Fast"
                    : quality === "balanced"
                      ? "Balanced"
                      : quality === "high"
                        ? "High fidelity"
                        : "Full resolution"}
                </option>
              ))}
            </select>
            <small>{libraryQualitySummary}</small>
          </label>
        )}
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

        <article>
          <header>
            <h2>4. ArUco Segmentation vs SAM2</h2>
            <p>Detect ArUco markers along your object boundary, build a polygonal mask, and compare it against SAM2 predictions.</p>
          </header>
          <label className="field">
            <span>Segmentation dataset uploads</span>
            <input type="file" accept="image/*" multiple onChange={(event) => handleSegFiles(event.target.files)} />
            <small>{segFileSummary}</small>
          </label>
          <label className="field">
            <span>SAM2 mask bundle (optional)</span>
            <input
              type="file"
              accept=".zip,.npz,.npy,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
              onChange={(event) => setSamMaskBundle(event.target.files?.[0] ?? null)}
            />
            <small>{samMaskSummary}</small>
          </label>
          <form
            className="mini-form"
            onSubmit={(evt: FormEvent) => {
              evt.preventDefault();
              runSegmentation();
            }}
          >
            <label>
              ArUco dictionary
              <select value={dictionary} onChange={(e) => setDictionary(e.target.value)}>
                {[
                  "DICT_4X4_50",
                  "DICT_4X4_100",
                  "DICT_5X5_50",
                  "DICT_5X5_100",
                  "DICT_6X6_100",
                ].map((dict) => (
                  <option key={dict} value={dict}>
                    {dict}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Minimum markers
              <input type="number" min={3} value={minMarkers} onChange={(e) => setMinMarkers(Number(e.target.value))} />
            </label>
            <label>
              Dilate kernel (px)
              <input type="number" min={1} value={dilatePx} onChange={(e) => setDilatePx(Number(e.target.value))} />
            </label>
            <label>
              Smooth kernel (px)
              <input type="number" min={1} value={smoothPx} onChange={(e) => setSmoothPx(Number(e.target.value))} />
            </label>
            {sam2BackendEnabled ? (
              <>
                <label>
                  SAM2 model ID
                  <input type="text" value={samModelId} onChange={(e) => setSamModelId(e.target.value)} />
                </label>
                <label className="toggle-field">
                  <span>Compare against SAM2</span>
                  <input type="checkbox" checked={compareSam} onChange={(e) => setCompareSam(e.target.checked)} />
                  <small className="helper-text">{sam2StatusMessage}</small>
                </label>
              </>
            ) : (
              <p className="form-note field-hint--error">{sam2StatusMessage}</p>
            )}
            <button type="submit" className="primary-btn" disabled={segLoading}>
              {segLoading ? "Segmenting…" : segmentationButtonLabel}
            </button>
          </form>
          {segSummary && (
            <div className="metric-grid">
              <div className="metric-tile">
                <p>Total frames</p>
                <strong>{segSummary.total_images}</strong>
              </div>
              <div className="metric-tile">
                <p>Successful masks</p>
                <strong>{segSummary.success_count}</strong>
              </div>
              {segSummary.avg_marker_count && (
                <div className="metric-tile">
                  <p>Avg markers</p>
                  <strong>{segSummary.avg_marker_count.toFixed(2)}</strong>
                </div>
              )}
              {segSummary.avg_sam_iou && (
                <div className="metric-tile">
                  <p>Avg IoU (SAM2)</p>
                  <strong>{(segSummary.avg_sam_iou * 100).toFixed(1)}%</strong>
                </div>
              )}
              {segSummary.avg_sam_dice && (
                <div className="metric-tile">
                  <p>Avg Dice</p>
                  <strong>{segSummary.avg_sam_dice.toFixed(3)}</strong>
                </div>
              )}
              {segSummary.summary_csv_url && (
                <div className="metric-tile">
                  <p>Summary CSV</p>
                  <strong>
                    <a href={segSummary.summary_csv_url} target="_blank" rel="noreferrer" className="download-link">
                      Download
                    </a>
                  </strong>
                </div>
              )}
              {segSummary.uploaded_mask_matches && (
                <div className="metric-tile">
                  <p>Uploaded SAM masks</p>
                  <strong>{segSummary.uploaded_mask_matches}</strong>
                </div>
              )}
            </div>
          )}
          {segResults && (
            <div className="result-list">
              {segResults.map((item) => (
                <div key={item.image} className="result-card">
                  <h3>{item.image}</h3>
                  <p>Markers: {item.marker_count}</p>
                  {item.coverage_pct && <p>Coverage: {item.coverage_pct.toFixed(1)}%</p>}
                  {item.sam_iou != null && <p>SAM2 IoU: {(item.sam_iou * 100).toFixed(1)}%</p>}
                  {item.sam_dice != null && <p>SAM2 Dice: {item.sam_dice.toFixed(3)}</p>}
                  {item.sam_quality != null && <p>SAM2 Quality: {item.sam_quality.toFixed(3)}</p>}
                  {item.issues && <p className="form-error">{item.issues}</p>}
                  {item.sam_error && <p className="form-error">SAM2: {item.sam_error}</p>}
                  <div className="image-strip">
                    {item.overlay_url && <img src={item.overlay_url} alt={`${item.image} ArUco`} />}
                    {item.sam_overlay_url && <img src={item.sam_overlay_url} alt={`${item.image} SAM2`} />}
                    {item.markers_debug_url && <img src={item.markers_debug_url} alt={`${item.image} markers`} />}
                  </div>
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
