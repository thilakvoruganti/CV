import { FormEvent, useMemo, useState } from "react";
import { api } from "../api/client";
import ImageModal from "../components/ImageModal";
import "./ImageStitchingPage.css";

type StitchResponse = {
  ok: boolean;
  panorama_url: string;
  compare_url?: string | null;
  elapsed_sec?: number;
};

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
  const [feature, setFeature] = useState("sift");
  const [maxWidth, setMaxWidth] = useState(1400);
  const [fitMode, setFitMode] = useState<"fit" | "scroll">("fit");
  const [panoramaUrl, setPanoramaUrl] = useState<string | null>(null);
  const [compareUrl, setCompareUrl] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const exceedsMinRequirement = selectedFiles.length >= 4;

  const fileSummary = useMemo(() => {
    if (!selectedFiles.length) return "No files selected";
    const list = selectedFiles.map((f) => f.name).join(", ");
    return `${selectedFiles.length} image${selectedFiles.length > 1 ? "s" : ""}: ${list}`;
  }, [selectedFiles]);

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

    if (!exceedsMinRequirement) {
      setError("Please select at least 4 overlapping images.");
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("images", file));
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
            Upload at least four overlapping photos captured from a single vantage point. We will detect features using OpenCV, align every
            frame, and blend the result into a high-resolution panorama powered by the FastAPI backend.
          </p>
        </header>

        <form className="stitch-form" onSubmit={handleSubmit}>
          <label className="field">
            <span>Upload images (min 4)</span>
            <input type="file" accept="image/*" multiple onChange={(event) => handleFiles(event.target.files)} />
            <small className={exceedsMinRequirement ? "" : "field-hint--error"}>{fileSummary}</small>
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Max width (px)</span>
              <input
                type="number"
                min={800}
                max={2400}
                value={maxWidth}
                onChange={(event) => setMaxWidth(Number(event.target.value))}
              />
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
