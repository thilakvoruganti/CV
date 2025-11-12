import { FormEvent, useState } from "react";
import { api } from "../api/client";
import ImageModal from "../components/ImageModal";
import "./SiftComparePage.css";

type PipelineMode = "both" | "opencv" | "custom";

type SiftCompareResponse = {
  ok: boolean;
  pipelines: PipelineMode[];
  opencv_keypoints?: [number, number];
  opencv_matches?: number;
  opencv_inliers?: number;
  opencv_time?: number;
  custom_keypoints?: [number, number];
  custom_matches?: number;
  custom_inliers_count?: number;
  custom_time?: number;
  custom_inliers_url?: string;
  opencv_inliers_url?: string;
};

const booleanField = (value: boolean) => (value ? "true" : "false");

export function SiftComparePage() {
  const [imageA, setImageA] = useState<File | null>(null);
  const [imageB, setImageB] = useState<File | null>(null);
  const [targetHeight, setTargetHeight] = useState(1100);
  const [ratio, setRatio] = useState(0.85);
  const [ransacThresh, setRansacThresh] = useState(15);
  const [ransacIters, setRansacIters] = useState(9000);
  const [rootSift, setRootSift] = useState(true);
  const [crossCheck, setCrossCheck] = useState(false);
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>("both");
  const [result, setResult] = useState<SiftCompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  const handleSubmit = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    setError(null);

    if (!imageA || !imageB) {
      setError("Please choose two images to compare.");
      return;
    }

    const formData = new FormData();
    formData.append("image1", imageA);
    formData.append("image2", imageB);
    formData.append("target_height", String(targetHeight));
    formData.append("ratio", String(ratio));
    formData.append("ransac_thresh", String(ransacThresh));
    formData.append("ransac_iters", String(ransacIters));
    formData.append("root_sift", booleanField(rootSift));
    formData.append("cross_check", booleanField(crossCheck));
    formData.append("pipelines", pipelineMode);

    try {
      setLoading(true);
      const { data } = await api.post<SiftCompareResponse>("/api/sift_compare", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult({
        ...data,
        custom_inliers_url: data.custom_inliers_url ? new URL(data.custom_inliers_url, api.defaults.baseURL).toString() : undefined,
        opencv_inliers_url: data.opencv_inliers_url ? new URL(data.opencv_inliers_url, api.defaults.baseURL).toString() : undefined,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed. Please retry with different inputs.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="sift-root">
      <div className="sift-panel">
        <header>
          <p className="sift-eyebrow">Feature Comparison</p>
          <h1>SIFT Playground</h1>
          <p className="sift-subhead">
            Benchmark our handcrafted SIFT implementation against OpenCV’s reference. Upload two images, tweak match criteria, and inspect
            the inlier visualizations.
          </p>
        </header>

        <form className="sift-form" onSubmit={handleSubmit}>
          <label className="field">
            <span>Image A</span>
            <input type="file" accept="image/*" onChange={(event) => setImageA(event.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            <span>Image B</span>
            <input type="file" accept="image/*" onChange={(event) => setImageB(event.target.files?.[0] ?? null)} />
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Target height (px)</span>
              <input type="number" min={400} max={2000} value={targetHeight} onChange={(e) => setTargetHeight(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>Ratio test</span>
              <input type="number" step={0.01} min={0.1} max={0.99} value={ratio} onChange={(e) => setRatio(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>RANSAC threshold</span>
              <input type="number" step={0.5} min={0.5} max={15} value={ransacThresh} onChange={(e) => setRansacThresh(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>RANSAC iterations</span>
              <input type="number" min={100} max={20000} value={ransacIters} onChange={(e) => setRansacIters(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>Pipeline</span>
              <select value={pipelineMode} onChange={(e) => setPipelineMode(e.target.value as PipelineMode)}>
                <option value="both">Both (custom & OpenCV)</option>
                <option value="custom">Custom only</option>
                <option value="opencv">OpenCV only</option>
              </select>
            </label>
          </div>

          <div className="field-row">
            <label className="toggle-field">
              <input type="checkbox" checked={rootSift} onChange={(e) => setRootSift(e.target.checked)} />
              <span>Enable RootSIFT normalization</span>
            </label>
            <label className="toggle-field">
              <input type="checkbox" checked={crossCheck} onChange={(e) => setCrossCheck(e.target.checked)} />
              <span>Cross-check matches</span>
            </label>
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="primary-btn" disabled={loading}>
            {loading ? "Comparing..." : "Run Comparison"}
          </button>
        </form>
      </div>

      <div className="sift-results">
        <h2>Results</h2>
        {!result && <p className="result-placeholder">Upload two images and run the comparison to visualize keypoint matches.</p>}

        {result && (
          <>
            <div className="metrics-grid">
              {result.custom_keypoints && (
                <>
                  <div>
                    <p className="metric-label">Custom keypoints</p>
                    <p className="metric-value">
                      {result.custom_keypoints[0]} / {result.custom_keypoints[1]}
                    </p>
                  </div>
                  <div>
                    <p className="metric-label">Custom matches</p>
                    <p className="metric-value">{result.custom_matches}</p>
                  </div>
                  <div>
                    <p className="metric-label">Custom inliers</p>
                    <p className="metric-value">{result.custom_inliers_count}</p>
                  </div>
                  <div>
                    <p className="metric-label">Custom time</p>
                    <p className="metric-value">{result.custom_time}s</p>
                  </div>
                </>
              )}
              {result.opencv_keypoints && (
                <>
                  <div>
                    <p className="metric-label">OpenCV keypoints</p>
                    <p className="metric-value">
                      {result.opencv_keypoints[0]} / {result.opencv_keypoints[1]}
                    </p>
                  </div>
                  <div>
                    <p className="metric-label">OpenCV matches</p>
                    <p className="metric-value">{result.opencv_matches}</p>
                  </div>
                  <div>
                    <p className="metric-label">OpenCV inliers</p>
                    <p className="metric-value">{result.opencv_inliers}</p>
                  </div>
                  <div>
                    <p className="metric-label">OpenCV time</p>
                    <p className="metric-value">{result.opencv_time}s</p>
                  </div>
                </>
              )}
            </div>

            <div className="comparison-grid">
              {result.custom_inliers_url && (
                <figure>
                  <figcaption>Custom pipeline</figcaption>
                  <img
                    src={result.custom_inliers_url}
                    alt="Custom SIFT inliers"
                    loading="lazy"
                    role="button"
                    tabIndex={0}
                    onClick={() => setModalImage({ src: result.custom_inliers_url!, title: "Custom pipeline" })}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") setModalImage({ src: result.custom_inliers_url!, title: "Custom pipeline" });
                    }}
                  />
                </figure>
              )}
              {result.opencv_inliers_url && (
                <figure>
                  <figcaption>OpenCV SIFT</figcaption>
                  <img
                    src={result.opencv_inliers_url}
                    alt="OpenCV SIFT inliers"
                    loading="lazy"
                    role="button"
                    tabIndex={0}
                    onClick={() => setModalImage({ src: result.opencv_inliers_url!, title: "OpenCV SIFT" })}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") setModalImage({ src: result.opencv_inliers_url!, title: "OpenCV SIFT" });
                    }}
                  />
                </figure>
              )}
            </div>
          </>
        )}
      </div>
      <ImageModal open={Boolean(modalImage)} onClose={() => setModalImage(null)} title={modalImage?.title}>
        {modalImage && <img src={modalImage.src} alt={modalImage.title} />}
      </ImageModal>
    </section>
  );
}

export default SiftComparePage;
