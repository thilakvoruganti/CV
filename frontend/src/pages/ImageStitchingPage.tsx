import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import ImageModal from "../components/ImageModal";
import "./ImageStitchingPage.css";

type StitchResponse = {
  ok: boolean;
  panorama_url: string;
  compare_url?: string | null;
  elapsed_sec?: number;
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
};

const STITCH_MIN_IMAGES = 4;
const STITCH_MAX_UPLOAD_IMAGES = 12;
const STITCH_MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const STITCH_MAX_UPLOAD_SIZE_LABEL = `${(STITCH_MAX_UPLOAD_BYTES / (1024 * 1024)).toFixed(0)} MB`;
const STITCH_UPLOAD_LIMIT_SUMMARY = `${STITCH_MAX_UPLOAD_IMAGES} images (≤${STITCH_MAX_UPLOAD_SIZE_LABEL} each)`;
const STITCH_MIN_WIDTH = 800;
const STITCH_MAX_WIDTH = 2200;

const resolveStaticUrl = (path: string) => {
  const fallbackBase = typeof window !== "undefined" ? window.location.origin : "/";
  const base = api.defaults.baseURL ?? fallbackBase;
  try {
    return new URL(path, base).toString();
  } catch {
    return path;
  }
};

export function ImageStitchingPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [imageSource, setImageSource] = useState<"upload" | "module3">("upload");
  const [datasetEntries, setDatasetEntries] = useState<DatasetEntry[]>([]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [feature, setFeature] = useState("sift");
  const [maxWidth, setMaxWidth] = useState(1400);
  const [fitMode, setFitMode] = useState<"fit" | "scroll">("fit");
  const [panoramaUrl, setPanoramaUrl] = useState<string | null>(null);
  const [compareUrl, setCompareUrl] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const meetsRequirement = imageSource === "module3"
    ? datasetEntries.length >= STITCH_MIN_IMAGES
    : selectedFiles.length >= STITCH_MIN_IMAGES;

  const datasetSummary = useMemo(() => {
    if (datasetLoading) {
      return "Loading shared dataset...";
    }
    if (!datasetEntries.length) {
      return "No dataset stored yet.";
    }
    if (datasetEntries.length === 1) {
      return "1 stored image ready.";
    }
    return `${datasetEntries.length} stored images ready (up to ${STITCH_MAX_UPLOAD_IMAGES} used per run).`;
  }, [datasetEntries, datasetLoading]);

  const fileSummary = useMemo(() => {
    if (!selectedFiles.length) {
      return `No files selected. Limit ${STITCH_UPLOAD_LIMIT_SUMMARY}.`;
    }
    const list = selectedFiles.map((f) => f.name).join(", ");
    return `${selectedFiles.length} image${selectedFiles.length > 1 ? "s" : ""}: ${list} • limit ${STITCH_UPLOAD_LIMIT_SUMMARY}.`;
  }, [selectedFiles]);

  const fetchDataset = useCallback(async () => {
    setDatasetLoading(true);
    try {
      const { data } = await api.get<DatasetListResponse>("/api/module3/dataset");
      setDatasetEntries(data.entries ?? []);
      setDatasetError(null);
    } catch (err) {
      setDatasetError(err instanceof Error ? err.message : "Unable to load dataset library.");
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDataset();
  }, [fetchDataset]);

  const handleFiles = (files: FileList | null) => {
    if (!files) {
      setSelectedFiles([]);
      return;
    }
    const nextFiles = Array.from(files).filter((file) => file.type.startsWith("image/"));
    setSelectedFiles(nextFiles);
  };

  const handleSubmit = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    setError(null);

    if (imageSource === "module3") {
      if (!datasetEntries.length) {
        setError("Shared dataset is empty. Upload frames there first.");
        return;
      }
      if (datasetEntries.length < STITCH_MIN_IMAGES) {
        setError(`Shared dataset needs at least ${STITCH_MIN_IMAGES} images.`);
        return;
      }
    } else if (selectedFiles.length < STITCH_MIN_IMAGES) {
      setError(`Please select at least ${STITCH_MIN_IMAGES} overlapping images.`);
      return;
    }

    if (imageSource === "upload") {
      if (selectedFiles.length > STITCH_MAX_UPLOAD_IMAGES) {
        setError(`Upload up to ${STITCH_MAX_UPLOAD_IMAGES} images per run.`);
        return;
      }
      const oversize = selectedFiles.find((file) => file.size > STITCH_MAX_UPLOAD_BYTES);
      if (oversize) {
        setError(`${oversize.name} exceeds the ${STITCH_MAX_UPLOAD_SIZE_LABEL} limit.`);
        return;
      }
    }

    const formData = new FormData();
    formData.append("image_source", imageSource);
    if (imageSource === "upload") {
      selectedFiles.forEach((file) => formData.append("images", file));
    }
    formData.append("feature", feature);
    formData.append("max_width", String(maxWidth));

    try {
      setIsSubmitting(true);
      const { data } = await api.post<StitchResponse>("/api/stitch", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPanoramaUrl(data.panorama_url ? resolveStaticUrl(data.panorama_url) : null);
      setCompareUrl(data.compare_url ? resolveStaticUrl(data.compare_url) : null);
      setElapsed(data.elapsed_sec ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stitch images. Please try again.");
      setPanoramaUrl(null);
      setCompareUrl(null);
      setElapsed(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="stitch-root">
      <div className="stitch-panel">
        <header>
          <p className="stitch-eyebrow">Panorama Lab</p>
          <h1>Image Stitching</h1>
          <p className="stitch-subhead">
            Upload between {STITCH_MIN_IMAGES} and {STITCH_MAX_UPLOAD_IMAGES} overlapping photos (≤{STITCH_MAX_UPLOAD_SIZE_LABEL} each) captured from a
            single vantage point. We detect features with OpenCV, align every frame, and blend the result into a high-resolution panorama powered by
            the FastAPI backend.
          </p>
        </header>

        <form className="stitch-form" onSubmit={handleSubmit}>
          <label className="field">
            <span>Image Source</span>
            <select value={imageSource} onChange={(event) => setImageSource(event.target.value as "upload" | "module3")}>
              <option value="upload">Upload from this device</option>
              <option value="module3" disabled={!datasetEntries.length}>
                Shared dataset {datasetEntries.length ? `(${datasetEntries.length} images)` : "(empty)"}
              </option>
            </select>
            <small>{imageSource === "module3" ? datasetSummary : "Files stay local until you submit."}</small>
            {datasetError && <small className="field-hint--error">{datasetError}</small>}
            {imageSource === "module3" && !datasetLoading && !datasetEntries.length && (
              <small className="field-hint--error">Upload dataset images in Edge Lab first.</small>
            )}
          </label>

          <label className="field">
            <span>Upload images (min 4)</span>
            <input
              type="file"
              accept="image/*"
              multiple
              disabled={imageSource === "module3"}
              onChange={(event) => handleFiles(event.target.files)}
            />
            <small className={imageSource === "upload" && !meetsRequirement ? "field-hint--error" : ""}>
              {imageSource === "module3" ? "Uploads are disabled when using the shared dataset." : fileSummary}
            </small>
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Max width (px)</span>
              <input
                type="number"
                min={STITCH_MIN_WIDTH}
                max={STITCH_MAX_WIDTH}
                value={maxWidth}
                onChange={(event) => setMaxWidth(Number(event.target.value))}
              />
              <small>Backend clamps this between {STITCH_MIN_WIDTH}px and {STITCH_MAX_WIDTH}px.</small>
            </label>
            <label className="field">
              <span>Feature Detector</span>
              <select value={feature} onChange={(event) => setFeature(event.target.value)}>
                <option value="sift">SIFT</option>
                <option value="orb">ORB</option>
              </select>
            </label>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="primary-btn" disabled={isSubmitting}>
            {isSubmitting ? "Stitching..." : "Build Panorama"}
          </button>
        </form>
      </div>

      <div className="stitch-results">
        <h2>Output</h2>
        {!panoramaUrl && <p className="result-placeholder">Your panorama preview will appear here once the backend finishes processing.</p>}

        {panoramaUrl && (
          <>
            <div className="result-toolbar">
              <a className="primary-btn ghost" href={panoramaUrl} download target="_blank" rel="noreferrer">
                Download Panorama
              </a>
              {elapsed !== null && <span className="elapsed">⏱ {elapsed}s</span>}
              <div className="view-toggle">
                <button
                  type="button"
                  className={fitMode === "fit" ? "toggle-btn active" : "toggle-btn"}
                  onClick={() => setFitMode("fit")}
                >
                  Fit
                </button>
                <button
                  type="button"
                  className={fitMode === "scroll" ? "toggle-btn active" : "toggle-btn"}
                  onClick={() => setFitMode("scroll")}
                >
                  Actual
                </button>
              </div>
            </div>
            <div className={fitMode === "fit" ? "result-frame" : "result-frame scrollable"}>
              <img
                className={fitMode === "fit" ? "stitch-image stitch-image--fit" : "stitch-image stitch-image--scroll"}
                src={panoramaUrl}
                alt="Panorama output"
                role="button"
                tabIndex={0}
                onClick={() => setModalImage({ src: panoramaUrl, title: "Panorama output" })}
                onKeyDown={(event) => {
                  if (event.key === "Enter") setModalImage({ src: panoramaUrl, title: "Panorama output" });
                }}
              />
            </div>
            {compareUrl && (
              <div className="result-frame muted">
                <p className="compare-label">Side-by-side comparison</p>
                <img
                  src={compareUrl}
                  alt="Mobile comparison"
                  role="button"
                  tabIndex={0}
                  onClick={() => setModalImage({ src: compareUrl, title: "Side-by-side comparison" })}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") setModalImage({ src: compareUrl, title: "Side-by-side comparison" });
                  }}
                />
              </div>
            )}
          </>
        )}
      </div>
      <ImageModal open={Boolean(modalImage)} onClose={() => setModalImage(null)} title={modalImage?.title}>
        {modalImage && <img src={modalImage.src} alt={modalImage.title} />}
      </ImageModal>
    </section>
  );
}

export default ImageStitchingPage;
