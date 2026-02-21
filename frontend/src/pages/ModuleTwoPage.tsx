import { FormEvent, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import ImageModal from "../components/ImageModal";
import "./ModuleTwoPage.css";

type MatchResponse = {
  ok: boolean;
  template_name: string;
  best_score: number;
  scale: number;
  angle: number;
  threshold: number;
  location: { x: number; y: number; width: number; height: number };
  annotated_url: string;
};

type FourierResponse = {
  ok: boolean;
  sigma: number;
  ksize: number;
  wiener_k: number;
  psnr_blur: number;
  psnr_restore: number;
  original_url: string;
  blurred_url: string;
  restored_url: string;
};

type LibraryTemplate = {
  template: string;
  score?: number;
  scale?: number;
  angle?: number;
  top_left?: { x: number; y: number };
  size?: { width: number; height: number };
  passed?: boolean;
  error?: string;
};

type LibraryResponse = {
  ok: boolean;
  templates: LibraryTemplate[];
  matches: LibraryTemplate[];
  threshold: number;
  annotated_url?: string | null;
  blurred_url?: string | null;
  total_templates?: number;
  processed_templates?: number;
  max_templates?: number | null;
  limited?: boolean;
  skipped_templates?: number;
};

type LibraryUploadResponse = {
  ok: boolean;
  saved: { original: string; stored_as: string; size: number }[];
  total_templates: number;
};

type TemplateInfo = {
  name: string;
  size: number;
  modified: number;
};

type TemplateListResponse = {
  ok: boolean;
  templates: TemplateInfo[];
  count: number;
};

const toAbsolute = (path?: string | null) => {
  if (!path) return path ?? "";
  try {
    return path.startsWith("http") ? path : new URL(path, api.defaults.baseURL).toString();
  } catch {
    return path;
  }
};

export function ModuleTwoPage() {
  // Section 1 – template match
  const [sceneFile, setSceneFile] = useState<File | null>(null);
  const [templateFile, setTemplateFile] = useState<File | null>(null);
  const [scoreThresh, setScoreThresh] = useState(0.7);
  const [scaleMin, setScaleMin] = useState(0.6);
  const [scaleMax, setScaleMax] = useState(1.3);
  const [scaleSteps, setScaleSteps] = useState(12);
  const [allowFlip, setAllowFlip] = useState(true);
  const [matchResult, setMatchResult] = useState<MatchResponse | null>(null);
  const [matchError, setMatchError] = useState<string | null>(null);
  const [matchLoading, setMatchLoading] = useState(false);

  // Section 2 – Fourier
  const [fourierImage, setFourierImage] = useState<File | null>(null);
  const [sigma, setSigma] = useState(4);
  const [displayWidth, setDisplayWidth] = useState(1024);
  const [wienerK, setWienerK] = useState<number | "">("");
  const [fourierResult, setFourierResult] = useState<FourierResponse | null>(null);
  const [fourierError, setFourierError] = useState<string | null>(null);
  const [fourierLoading, setFourierLoading] = useState(false);

  // Section 3 – library
  const [libraryScene, setLibraryScene] = useState<File | null>(null);
  const [libraryThresh, setLibraryThresh] = useState(0.7);
  const [libraryBlurKernel, setLibraryBlurKernel] = useState(35);
  const [libraryAllowFlip, setLibraryAllowFlip] = useState(true);
  const [libraryResult, setLibraryResult] = useState<LibraryResponse | null>(null);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(false);

  // Section 0 – template uploads
  const [templateFiles, setTemplateFiles] = useState<File[]>([]);
  const [templateUploadMessage, setTemplateUploadMessage] = useState<string | null>(null);
  const [templateUploadError, setTemplateUploadError] = useState<string | null>(null);
  const [templateUploadLoading, setTemplateUploadLoading] = useState(false);
  const [templateResetLoading, setTemplateResetLoading] = useState(false);
  const [templateList, setTemplateList] = useState<TemplateInfo[]>([]);
  const [templateListLoading, setTemplateListLoading] = useState(false);
  const [matchTemplateSource, setMatchTemplateSource] = useState<"upload" | "library">("upload");
  const [selectedLibraryTemplate, setSelectedLibraryTemplate] = useState("");

  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const fetchTemplateList = async () => {
    setTemplateListLoading(true);
    try {
      const { data } = await api.get<TemplateListResponse>("/api/object/library_templates");
      setTemplateList(data.templates ?? []);
    } catch (err) {
      console.error(err);
      setTemplateUploadError(err instanceof Error ? err.message : "Unable to load template library.");
    } finally {
      setTemplateListLoading(false);
    }
  };

  useEffect(() => {
    void fetchTemplateList();
  }, []);

  const sceneSummary = useMemo(() => (sceneFile ? `${sceneFile.name}` : "Select a scene image"), [sceneFile]);
  const templateSummary = useMemo(() => (templateFile ? `${templateFile.name}` : "Select a template image"), [templateFile]);
  const fourierSummary = useMemo(() => (fourierImage ? `${fourierImage.name}` : "Select an image to blur"), [fourierImage]);
  const librarySceneSummary = useMemo(() => (libraryScene ? `${libraryScene.name}` : "Select a scene to scan"), [libraryScene]);
  const templateUploadSummary = useMemo(() => {
    if (templateFiles.length === 0) {
      return "Select template crops captured from different scenes.";
    }
    if (templateFiles.length === 1) {
      return templateFiles[0].name;
    }
    return `${templateFiles.length} files ready (${templateFiles.map((file) => file.name).join(", ")})`;
  }, [templateFiles]);
  const libraryTemplateCount = templateList.length;
  const hasLibraryTemplates = libraryTemplateCount > 0;

  const submitTemplateUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (templateFiles.length === 0) {
      setTemplateUploadError("Select at least one template image.");
      return;
    }

    const formData = new FormData();
    templateFiles.forEach((file) => formData.append("templates", file));

    try {
      setTemplateUploadLoading(true);
      setTemplateUploadError(null);
      const { data } = await api.post<LibraryUploadResponse>("/api/object/library_upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setTemplateUploadMessage(
        `Uploaded ${data.saved.length} template${data.saved.length === 1 ? "" : "s"}. Total stored: ${data.total_templates}.`,
      );
      setTemplateFiles([]);
      form.reset();
      await fetchTemplateList();
    } catch (err) {
      setTemplateUploadError(err instanceof Error ? err.message : "Template upload failed.");
      setTemplateUploadMessage(null);
    } finally {
      setTemplateUploadLoading(false);
    }
  };

  const handleTemplateReset = async () => {
    if (!window.confirm("Delete all stored templates? This cannot be undone.")) {
      return;
    }
    try {
      setTemplateResetLoading(true);
      setTemplateUploadError(null);
      setTemplateUploadMessage(null);
      await api.delete("/api/object/library_templates");
      setTemplateUploadMessage("Template library cleared.");
      setTemplateFiles([]);
      setSelectedLibraryTemplate("");
      setMatchTemplateSource("upload");
      await fetchTemplateList();
    } catch (err) {
      setTemplateUploadError(err instanceof Error ? err.message : "Unable to reset template library.");
      setTemplateUploadMessage(null);
    } finally {
      setTemplateResetLoading(false);
    }
  };

  const submitMatch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!sceneFile) {
      setMatchError("Upload a scene image.");
      return;
    }

    const formData = new FormData();
    formData.append("scene", sceneFile);
    let endpoint = "/api/object/match";

    if (matchTemplateSource === "upload") {
      if (!templateFile) {
        setMatchError("Upload a template image or switch to the library option.");
        return;
      }
      formData.append("template", templateFile);
    } else {
      if (!selectedLibraryTemplate) {
        setMatchError("Select a template from the library.");
        return;
      }
      formData.append("template_name", selectedLibraryTemplate);
      endpoint = "/api/object/match/library";
    }
    formData.append("score_thresh", String(scoreThresh));
    formData.append("scale_min", String(scaleMin));
    formData.append("scale_max", String(scaleMax));
    formData.append("scale_steps", String(scaleSteps));
    formData.append("allow_flip", allowFlip ? "true" : "false");
    try {
      setMatchLoading(true);
      setMatchError(null);
      const { data } = await api.post<MatchResponse>(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMatchResult({
        ...data,
        annotated_url: toAbsolute(data.annotated_url),
      });
    } catch (err) {
      setMatchError(err instanceof Error ? err.message : "Template matching failed.");
      setMatchResult(null);
    } finally {
      setMatchLoading(false);
    }
  };

  const submitFourier = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!fourierImage) {
      setFourierError("Upload an image to process.");
      return;
    }
    const formData = new FormData();
    formData.append("image", fourierImage);
    formData.append("sigma", String(sigma));
    formData.append("display_width", String(displayWidth));
    if (wienerK !== "" && !Number.isNaN(wienerK)) {
      formData.append("wiener_k", String(wienerK));
    }
    try {
      setFourierLoading(true);
      setFourierError(null);
      const { data } = await api.post<FourierResponse>("/api/object/fourier", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setFourierResult({
        ...data,
        original_url: toAbsolute(data.original_url),
        blurred_url: toAbsolute(data.blurred_url),
        restored_url: toAbsolute(data.restored_url),
      });
    } catch (err) {
      setFourierError(err instanceof Error ? err.message : "Fourier filtering failed.");
      setFourierResult(null);
    } finally {
      setFourierLoading(false);
    }
  };

  const submitLibrary = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!libraryScene) {
      setLibraryError("Upload a scene image.");
      return;
    }
    const formData = new FormData();
    formData.append("scene", libraryScene);
    formData.append("score_thresh", String(libraryThresh));
    formData.append("blur_kernel", String(libraryBlurKernel));
    formData.append("allow_flip", libraryAllowFlip ? "true" : "false");
    try {
      setLibraryLoading(true);
      setLibraryError(null);
      const { data } = await api.post<LibraryResponse>("/api/object/library_detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLibraryResult({
        ...data,
        annotated_url: data.annotated_url ? toAbsolute(data.annotated_url) : undefined,
        blurred_url: data.blurred_url ? toAbsolute(data.blurred_url) : undefined,
      });
    } catch (err) {
      setLibraryError(err instanceof Error ? err.message : "Library lookup failed.");
      setLibraryResult(null);
    } finally {
      setLibraryLoading(false);
    }
  };

  return (
    <section className="module2-root">
      <header className="module2-header">
        <p className="eyebrow">Template Matching Lab</p>
        <h1>Correlation-based detection & Fourier filtering</h1>
        <p>
          Experiment with zero-mean normalized correlation for object detection, evaluate Gaussian blur vs. Fourier restoration, and run a
          local template database that auto-blurs detected regions.
        </p>
      </header>

      <div className="module2-grid">
        <article className="module2-card">
          <header>
            <h2>0. Template Library Manager</h2>
            <p>
              Upload at least ten exemplar crops (PNG/JPG/WebP) taken from different scenes. Each file must be under 100 KB to keep Render within
              free-tier limits and power the library detector below.
            </p>
          </header>
          <form className="module2-form" onSubmit={submitTemplateUpload}>
            <label className="field">
              <span>Template files</span>
              <input
                type="file"
                accept="image/png, image/jpeg, image/webp"
                multiple
                onChange={(event) => setTemplateFiles(Array.from(event.target.files ?? []))}
              />
              <small>{templateUploadSummary}</small>
            </label>
            {templateUploadError && <p className="form-error">{templateUploadError}</p>}
            {templateUploadMessage && <p className="form-note">{templateUploadMessage}</p>}
            <button type="submit" className="secondary-btn" disabled={templateUploadLoading}>
              {templateUploadLoading ? "Uploading…" : "Upload templates"}
            </button>
          </form>
          <div className="template-manager-actions">
            <button
              type="button"
              className="ghost-btn"
              onClick={() => void fetchTemplateList()}
              disabled={templateListLoading}
            >
              {templateListLoading ? "Refreshing…" : "Refresh list"}
            </button>
            <button
              type="button"
              className="danger-btn"
              onClick={handleTemplateReset}
              disabled={templateResetLoading || !hasLibraryTemplates}
            >
              {templateResetLoading ? "Clearing…" : "Reset library"}
            </button>
          </div>
          <p className="helper-text">
            Stored templates: {libraryTemplateCount} (need 10+ unique exemplars for robust library matching).
          </p>
          <p className="helper-text">Detection scans process the first 40 templates per run to keep Render responsive.</p>
          {hasLibraryTemplates ? (
            <div className="template-library-list">
              {templateList.slice(0, 8).map((tpl) => (
                <div key={tpl.name} className="template-row">
                  <div className="template-name">{tpl.name}</div>
                  <div className="template-meta">
                    {(tpl.size / 1024).toFixed(1)} KB · {new Date(tpl.modified * 1000).toLocaleDateString()}
                  </div>
                </div>
              ))}
              {libraryTemplateCount > 8 && <p className="helper-text">{libraryTemplateCount - 8}+ additional templates stored.</p>}
            </div>
          ) : (
            <p className="result-placeholder">No templates stored yet. Upload crops from scenes different than your test images.</p>
          )}
        </article>

        <article className="module2-card">
          <header>
            <h2>1. Template Matching (Correlation)</h2>
            <p>Upload a scene and a template cropped from a different capture. We’ll sweep scales and orientations to find the best match.</p>
          </header>
          <form className="module2-form" onSubmit={submitMatch}>
            <label className="field">
              <span>Scene image</span>
              <input type="file" accept="image/*" onChange={(event) => setSceneFile(event.target.files?.[0] ?? null)} />
              <small>{sceneSummary}</small>
            </label>
            <div className="field">
              <span>Template source</span>
              <div className="template-source-options">
                <label>
                  <input
                    type="radio"
                    name="template-source"
                    value="upload"
                    checked={matchTemplateSource === "upload"}
                    onChange={() => setMatchTemplateSource("upload")}
                  />
                  Upload new template
                </label>
                <label className={!hasLibraryTemplates ? "disabled" : undefined}>
                  <input
                    type="radio"
                    name="template-source"
                    value="library"
                    checked={matchTemplateSource === "library"}
                    onChange={() => setMatchTemplateSource("library")}
                    disabled={!hasLibraryTemplates}
                  />
                  Use template library ({libraryTemplateCount})
                </label>
              </div>
            </div>
            {matchTemplateSource === "upload" ? (
              <label className="field">
                <span>Template image</span>
                <input type="file" accept="image/*" onChange={(event) => setTemplateFile(event.target.files?.[0] ?? null)} />
                <small>{templateSummary}</small>
              </label>
            ) : (
              <label className="field">
                <span>Library template</span>
                <select
                  value={selectedLibraryTemplate}
                  onChange={(event) => setSelectedLibraryTemplate(event.target.value)}
                  disabled={!hasLibraryTemplates}
                >
                  <option value="">Select a stored template</option>
                  {templateList.map((tpl) => (
                    <option key={tpl.name} value={tpl.name}>
                      {tpl.name}
                    </option>
                  ))}
                </select>
                <small>
                  {hasLibraryTemplates
                    ? "Templates uploaded via the manager can be reused here."
                    : "Upload templates above to enable this option."}
                </small>
              </label>
            )}
            <div className="field-grid">
              <label className="field">
                <span>Min scale</span>
                <input type="number" step={0.05} min={0.2} value={scaleMin} onChange={(event) => setScaleMin(Number(event.target.value))} />
              </label>
              <label className="field">
                <span>Max scale</span>
                <input type="number" step={0.05} min={0.3} value={scaleMax} onChange={(event) => setScaleMax(Number(event.target.value))} />
              </label>
              <label className="field">
                <span># of scales</span>
                <input
                  type="number"
                  min={1}
                  max={25}
                  value={scaleSteps}
                  onChange={(event) => setScaleSteps(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Score threshold</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={scoreThresh}
                  onChange={(event) => setScoreThresh(Number(event.target.value))}
                />
              </label>
            </div>
            <label className="toggle-field">
              <input type="checkbox" checked={allowFlip} onChange={(event) => setAllowFlip(event.target.checked)} />
              <span>Check 180° flips</span>
            </label>
            {matchError && <p className="form-error">{matchError}</p>}
            <button type="submit" className="primary-btn" disabled={matchLoading}>
              {matchLoading ? "Matching…" : "Run template matching"}
            </button>
          </form>
          <p className="helper-text">Scales cap at 25 samples between 0.05× and 5× to keep correlation runs fast on Render.</p>
          {matchResult && (
            <div className="result-block">
              <div className="metrics-grid">
                <div>
                  <p className="metric-label">Template</p>
                  <p className="metric-value small">{matchResult.template_name}</p>
                </div>
                <div>
                  <p className="metric-label">Score</p>
                  <p className="metric-value">{matchResult.best_score.toFixed(3)}</p>
                </div>
                <div>
                  <p className="metric-label">Scale</p>
                  <p className="metric-value">{matchResult.scale.toFixed(2)}x</p>
                </div>
                <div>
                  <p className="metric-label">Angle</p>
                  <p className="metric-value">{matchResult.angle.toFixed(1)}°</p>
                </div>
              </div>
              <p className="metric-note">
                Bounding box: ({matchResult.location.x}, {matchResult.location.y}) →
                ({matchResult.location.width}×{matchResult.location.height}) pixels
              </p>
              <div className="annotated">
                <img
                  src={matchResult.annotated_url}
                  alt="Correlation detection"
                  onClick={() => setModalImage({ src: matchResult.annotated_url, title: "Correlation detection" })}
                />
                <button
                  className="ghost-btn"
                  type="button"
                  onClick={() => setModalImage({ src: matchResult.annotated_url, title: "Correlation detection" })}
                >
                  Expand
                </button>
              </div>
            </div>
          )}
        </article>

        <article className="module2-card">
          <header>
            <h2>2. Gaussian Blur + Fourier Recovery</h2>
            <p>Apply controllable Gaussian blur and attempt to recover the source via Wiener deconvolution in the frequency domain.</p>
          </header>
          <form className="module2-form" onSubmit={submitFourier}>
            <label className="field">
              <span>Input image</span>
              <input type="file" accept="image/*" onChange={(event) => setFourierImage(event.target.files?.[0] ?? null)} />
              <small>{fourierSummary}</small>
            </label>
            <div className="field-grid">
              <label className="field">
                <span>σ (blur)</span>
                <input type="number" step={0.5} min={0.5} value={sigma} onChange={(event) => setSigma(Number(event.target.value))} />
              </label>
              <label className="field">
                <span>Display width (px)</span>
                <input
                  type="number"
                  min={256}
                  max={1920}
                  value={displayWidth}
                  onChange={(event) => setDisplayWidth(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Wiener K (optional)</span>
                <input
                  type="number"
                  step={0.001}
                  placeholder="Auto"
                  value={wienerK}
                  onChange={(event) => {
                    const value = event.target.value;
                    setWienerK(value === "" ? "" : Number(value));
                  }}
                />
              </label>
            </div>
            {fourierError && <p className="form-error">{fourierError}</p>}
            <button type="submit" className="primary-btn" disabled={fourierLoading}>
              {fourierLoading ? "Processing…" : "Apply blur & recover"}
            </button>
          </form>
          {fourierResult && (
            <div className="result-block">
              <div className="metrics-grid">
                <div>
                  <p className="metric-label">σ / kernel</p>
                  <p className="metric-value">
                    {fourierResult.sigma.toFixed(2)} / {fourierResult.ksize}
                  </p>
                </div>
                <div>
                  <p className="metric-label">Wiener K</p>
                  <p className="metric-value">{fourierResult.wiener_k.toFixed(4)}</p>
                </div>
                <div>
                  <p className="metric-label">PSNR (blur)</p>
                  <p className="metric-value">{fourierResult.psnr_blur.toFixed(2)} dB</p>
                </div>
                <div>
                  <p className="metric-label">PSNR (restored)</p>
                  <p className="metric-value">{fourierResult.psnr_restore.toFixed(2)} dB</p>
                </div>
              </div>
              <div className="image-row">
                <figure>
                  <figcaption>Original</figcaption>
                  <img
                    src={fourierResult.original_url}
                    alt="Original"
                    onClick={() => setModalImage({ src: fourierResult.original_url, title: "Original" })}
                  />
                </figure>
                <figure>
                  <figcaption>Blurred</figcaption>
                  <img
                    src={fourierResult.blurred_url}
                    alt="Blurred"
                    onClick={() => setModalImage({ src: fourierResult.blurred_url, title: "Blurred" })}
                  />
                </figure>
                <figure>
                  <figcaption>Restored</figcaption>
                  <img
                    src={fourierResult.restored_url}
                    alt="Restored"
                    onClick={() => setModalImage({ src: fourierResult.restored_url, title: "Restored" })}
                  />
                </figure>
              </div>
            </div>
          )}
        </article>

        <article className="module2-card">
          <header>
            <h2>3. Library Detection & Blurring</h2>
            <p>Scan a scene against your local database of templates (10 objects recommended). Detected regions are blurred automatically.</p>
          </header>
          <form className="module2-form" onSubmit={submitLibrary}>
            <label className="field">
              <span>Scene image</span>
              <input type="file" accept="image/*" onChange={(event) => setLibraryScene(event.target.files?.[0] ?? null)} />
              <small>{librarySceneSummary}</small>
            </label>
            <div className="field-grid">
              <label className="field">
                <span>Score threshold</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={libraryThresh}
                  onChange={(event) => setLibraryThresh(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Blur kernel</span>
                <input
                  type="number"
                  min={3}
                  max={101}
                  step={2}
                  value={libraryBlurKernel}
                  onChange={(event) => setLibraryBlurKernel(Number(event.target.value))}
                />
              </label>
            </div>
            <label className="toggle-field">
              <input type="checkbox" checked={libraryAllowFlip} onChange={(event) => setLibraryAllowFlip(event.target.checked)} />
              <span>Allow 180° flips</span>
            </label>
            {libraryError && <p className="form-error">{libraryError}</p>}
            <button type="submit" className="primary-btn" disabled={libraryLoading}>
              {libraryLoading ? "Searching…" : "Find & blur objects"}
            </button>
          </form>
          {libraryResult && (
            <div className="result-block">
              {libraryResult.templates.length === 0 ? (
                <p className="result-placeholder">
                  No templates were found in your library. Upload at least ten exemplar crops via the Template Library Manager above to power this
                  demo.
                </p>
              ) : (
                <>
                  <p className="library-summary">
                    Processed {(libraryResult.processed_templates ?? libraryResult.templates.length).toLocaleString()} templates · {libraryResult.matches.length} passed the threshold ({libraryResult.threshold.toFixed(2)})
                  </p>
                  {libraryResult.limited && (
                    <p className="helper-text">
                      Showing the first {(libraryResult.processed_templates ?? libraryResult.templates.length).toLocaleString()} of
                      {" "}
                      {(libraryResult.total_templates ?? libraryResult.templates.length).toLocaleString()} stored templates (cap:
                      {" "}
                      {libraryResult.max_templates ?? "—"}).
                    </p>
                  )}
                  <div className="template-list">
                    {libraryResult.templates.map((tpl) => (
                      <div key={tpl.template} className={tpl.passed ? "template-row passed" : "template-row"}>
                        <div>
                          <strong>{tpl.template}</strong>
                          {tpl.error && <span className="template-error"> · {tpl.error}</span>}
                        </div>
                        {!tpl.error && (
                          <div className="template-meta">
                            score {tpl.score?.toFixed(2)} • scale {tpl.scale?.toFixed(2)}x • angle {tpl.angle?.toFixed(1)}°
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  {(libraryResult.annotated_url || libraryResult.blurred_url) && (
                    <div className="image-row compact">
                      {libraryResult.annotated_url && (
                        <figure>
                          <figcaption>Detections</figcaption>
                          <img
                            src={libraryResult.annotated_url}
                            alt="Library detections"
                            onClick={() => setModalImage({ src: libraryResult.annotated_url!, title: "Library detections" })}
                          />
                        </figure>
                      )}
                      {libraryResult.blurred_url && (
                        <figure>
                          <figcaption>Blurred regions</figcaption>
                          <img
                            src={libraryResult.blurred_url}
                            alt="Blurred objects"
                            onClick={() => setModalImage({ src: libraryResult.blurred_url!, title: "Blurred objects" })}
                          />
                        </figure>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </article>
      </div>

      <ImageModal open={Boolean(modalImage)} onClose={() => setModalImage(null)} title={modalImage?.title}>
        {modalImage && <img src={modalImage.src} alt={modalImage.title} />}
      </ImageModal>
    </section>
  );
}

export default ModuleTwoPage;
