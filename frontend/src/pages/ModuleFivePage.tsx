import axios from "axios";
import type { PointerEvent as ReactPointerEvent } from "react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, resolveApiUrl } from "../api/client";
import "./ModuleFivePage.css";

type Mode = "aruco" | "of" | "sam2";

type Roi = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type TrackerResponse = {
  session_id: string;
  frame_url: string;
  metadata: Record<string, unknown>;
  finished: boolean;
};

const TRACK_INTERVAL_MS = 250;
const FRAME_LIMIT_BYTES = 4 * 1024 * 1024;
const SAM2_LIMIT_BYTES = 8 * 1024 * 1024;
const TRACKER_LIMITS = {
  frameMaxDim: 960,
  sessionMaxFrames: 180,
  sessionTtlMinutes: 10,
} as const;

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

const frameLimitLabel = formatBytes(FRAME_LIMIT_BYTES);
const sam2LimitLabel = formatBytes(SAM2_LIMIT_BYTES);

const parseRoiPayload = (value: unknown): Roi | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Partial<Record<keyof Roi, unknown>>;
  const keys: Array<keyof Roi> = ["x", "y", "width", "height"];
  if (keys.every((key) => typeof candidate[key] === "number")) {
    return {
      x: candidate.x as number,
      y: candidate.y as number,
      width: candidate.width as number,
      height: candidate.height as number,
    };
  }
  return null;
};

export function ModuleFivePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const drawStateRef = useRef<{ startX: number; startY: number } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [mode, setMode] = useState<Mode>("aruco");
  const [markerId, setMarkerId] = useState<string>("");
  const [roi, setRoi] = useState<Roi | null>(null);
  const [roiOverlay, setRoiOverlay] = useState<Roi | null>(null);
  const [roiCaptureMode, setRoiCaptureMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [sending, setSending] = useState(false);
  const [sam2File, setSam2File] = useState<File | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  const [stageView, setStageView] = useState<"camera" | "tracked">("camera");

  const canDefineRoi = mode === "of" && videoReady;

  const stopInterval = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const stopStreaming = () => {
    stopInterval();
    setStreaming(false);
  };

  const stopCamera = () => {
    stopStreaming();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setVideoReady(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
      setError(null);
    } catch (err) {
      stopCamera();
      setError(err instanceof Error ? err.message : "Failed to access camera.");
    }
  };

  const captureFrameBlob = (): Promise<Blob> => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      return Promise.reject(new Error("Camera not ready. Allow access and wait a moment."));
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return Promise.reject(new Error("Cannot capture from camera."));
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Failed to capture frame."));
        } else {
          resolve(blob);
        }
      }, "image/jpeg", 0.9);
    });
  };

  const resetSession = () => {
    stopStreaming();
    setSessionId(null);
    setFrameUrl(null);
    setMetadata({});
    setStatus(null);
  };

  const submitFrame = async () => {
    if (sending) return;
    if (mode === "of" && !roi && !sessionId) {
      setError("Select an ROI before starting optical flow tracking.");
      stopStreaming();
      return;
    }
    if (mode === "sam2" && !sam2File && !sessionId) {
      setError("Upload the SAM2 mask NPZ file before starting.");
      stopStreaming();
      return;
    }

    try {
      setSending(true);
      const blob = await captureFrameBlob();
      const form = new FormData();
      form.append("mode", mode);
      form.append("frame", blob, `frame-${Date.now()}.jpg`);
      if (sessionId) {
        form.append("session_id", sessionId);
      }
      if (mode === "aruco" && markerId.trim()) {
        form.append("marker_id", markerId.trim());
      }
      if (mode === "of" && roi) {
        form.append("roi", JSON.stringify(roi));
      }
      if (mode === "sam2" && !sessionId && sam2File) {
        form.append("sam2_masks", sam2File);
      }

      const { data } = await api.post<TrackerResponse>("/api/module5/track", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const finished = Boolean(data.finished);
      setSessionId(finished ? null : data.session_id);
      const resolvedFrame = resolveApiUrl(data.frame_url);
      const cacheKey = `t=${Date.now()}`;
      setFrameUrl(resolvedFrame.includes("?") ? `${resolvedFrame}&${cacheKey}` : `${resolvedFrame}?${cacheKey}`);
      setMetadata(data.metadata || {});
      if (mode === "of") {
        const bbox = parseRoiPayload(data.metadata?.bbox as unknown);
        if (bbox) {
          setRoi(bbox);
        }
      }
      setStatus(finished ? "Tracker finished" : "Tracking live");
      setError(null);
      if (finished) {
        stopStreaming();
      }
    } catch (err: unknown) {
      let detail: string | null = null;
      if (axios.isAxiosError(err)) {
        const payload = err.response?.data as { detail?: string } | undefined;
        detail = typeof payload?.detail === "string" ? payload.detail : null;
      }
      const fallback = err instanceof Error ? err.message : "Failed to send frame.";
      setError(detail ?? fallback);
      const normalizedDetail = (detail ?? "").toLowerCase();
      if (normalizedDetail.includes("unknown session")) {
        resetSession();
      } else {
        stopStreaming();
      }
    } finally {
      setSending(false);
    }
  };

  const startStreaming = async () => {
    if (!videoReady) {
      setError("Start the camera and wait for it to initialize.");
      return;
    }
    if (mode === "of" && !roi && !sessionId) {
      setError("Draw an ROI before starting optical flow tracking.");
      return;
    }
    if (mode === "sam2" && !sam2File && !sessionId) {
      setError("Upload the SAM2 masks NPZ file before streaming.");
      return;
    }
    setStreaming(true);
    await submitFrame();
    stopInterval();
    intervalRef.current = window.setInterval(() => {
      submitFrame();
    }, TRACK_INTERVAL_MS);
  };

  const handleModeChange = (nextMode: Mode) => {
    if (nextMode !== mode) {
      resetSession();
      setMode(nextMode);
    }
  };

  const handleSam2Input = (file: File | null) => {
    if (file && file.size > SAM2_LIMIT_BYTES) {
      setError(`SAM2 mask bundle must be under ${sam2LimitLabel}.`);
      setSam2File(null);
      return;
    }
    setSam2File(file);
    setError(null);
  };

  const beginRoiCapture = () => {
    if (!canDefineRoi) {
      setError("Start the camera before defining an ROI.");
      return;
    }
    setError(null);
    setRoi(null);
    setRoiOverlay(null);
    setRoiCaptureMode(true);
  };

  const syncOverlayFromRoi = useCallback(() => {
    if (!roi || !containerRef.current || !videoRef.current) return;
    const bounds = containerRef.current.getBoundingClientRect();
    const videoWidth = videoRef.current.videoWidth || bounds.width;
    const videoHeight = videoRef.current.videoHeight || bounds.height;
    const scaleX = bounds.width / videoWidth;
    const scaleY = bounds.height / videoHeight;
    setRoiOverlay({
      x: roi.x * scaleX,
      y: roi.y * scaleY,
      width: roi.width * scaleX,
      height: roi.height * scaleY,
    });
  }, [roi]);

  useEffect(() => {
    syncOverlayFromRoi();
  }, [roi, videoReady, syncOverlayFromRoi]);

  const handlePointerDown = (event: ReactPointerEvent) => {
    if (!roiCaptureMode || !containerRef.current || stageView !== "camera") return;
    const bounds = containerRef.current.getBoundingClientRect();
    const startX = event.clientX - bounds.left;
    const startY = event.clientY - bounds.top;
    drawStateRef.current = { startX, startY };
    setRoiOverlay({ x: startX, y: startY, width: 0, height: 0 });
  };

  const handlePointerMove = (event: ReactPointerEvent) => {
    if (!roiCaptureMode || !drawStateRef.current || !containerRef.current || stageView !== "camera") return;
    const bounds = containerRef.current.getBoundingClientRect();
    const currentX = event.clientX - bounds.left;
    const currentY = event.clientY - bounds.top;
    const start = drawStateRef.current;
    const width = currentX - start.startX;
    const height = currentY - start.startY;
    const boxX = width >= 0 ? start.startX : currentX;
    const boxY = height >= 0 ? start.startY : currentY;
    setRoiOverlay({ x: boxX, y: boxY, width: Math.abs(width), height: Math.abs(height) });
  };

  const handlePointerUp = () => {
    if (
      !roiCaptureMode ||
      !drawStateRef.current ||
      !containerRef.current ||
      !videoRef.current ||
      !roiOverlay ||
      stageView !== "camera"
    ) {
      setRoiCaptureMode(false);
      drawStateRef.current = null;
      return;
    }
    const bounds = containerRef.current.getBoundingClientRect();
    const videoWidth = videoRef.current.videoWidth || bounds.width;
    const videoHeight = videoRef.current.videoHeight || bounds.height;
    const scaleX = videoWidth / bounds.width;
    const scaleY = videoHeight / bounds.height;

    const clamp = (value: number, max: number) => Math.min(Math.max(value, 0), max);

    const x = clamp(Math.round(roiOverlay.x * scaleX), videoWidth - 1);
    const y = clamp(Math.round(roiOverlay.y * scaleY), videoHeight - 1);
    const width = clamp(Math.round(roiOverlay.width * scaleX), videoWidth - x);
    const height = clamp(Math.round(roiOverlay.height * scaleY), videoHeight - y);

    setRoi({ x, y, width, height });
    setRoiCaptureMode(false);
    drawStateRef.current = null;
  };

  const metaEntries = useMemo(() => Object.entries(metadata || {}), [metadata]);

  const handleManualCapture = async (event: FormEvent) => {
    event.preventDefault();
    await submitFrame();
  };

  return (
    <section className="module5-root">
      <header className="module5-header">
        <p className="eyebrow">Tracking Lab</p>
        <h1>Hybrid Object Tracking Playground</h1>
        <p>Stream live camera frames into marker, markerless optical flow, or SAM2 segmentation trackers to showcase real-time behaviour.</p>
      </header>

      <div className="module5-guardrail">
        <strong>Render-safe guardrails</strong>
        <p>
          Frames are clamped to {TRACKER_LIMITS.frameMaxDim}px max dimension and {frameLimitLabel} per upload. Sessions auto-finish after {TRACKER_LIMITS.sessionMaxFrames} frames
          or roughly {TRACKER_LIMITS.sessionTtlMinutes} minutes of inactivity, and SAM2 bundles stay under {sam2LimitLabel}. Tracker outputs flush hourly to keep the Render disk tidy.
        </p>
      </div>

      {error && <p className="form-error">{error}</p>}
      {status && <p className="status-line">{status}</p>}

      <div className="module5-grid">
        <article>
          <h2>Camera Feed</h2>
          <div
            className={`camera-stage ${roiCaptureMode ? "camera-stage--capturing" : ""} ${stageView === "tracked" ? "camera-stage--tracked" : ""}`}
            ref={containerRef}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              onLoadedMetadata={() => setVideoReady(true)}
              className={`camera-feed ${stageView === "tracked" && frameUrl ? "camera-feed--hidden" : ""}`}
            />
            {stageView === "camera" && roiOverlay && (
              <div
                className="roi-overlay"
                style={{
                  left: roiOverlay.x,
                  top: roiOverlay.y,
                  width: roiOverlay.width,
                  height: roiOverlay.height,
                }}
              />
            )}
            {stageView === "tracked" && frameUrl && (
              <img src={frameUrl} alt="Tracked frame" className="tracker-stage-frame" />
            )}
          </div>
          <div className="camera-actions">
            <button type="button" className="secondary-btn" onClick={startCamera}>
              Enable Camera
            </button>
            <button type="button" className="secondary-btn" onClick={stopCamera}>
              Stop Camera
            </button>
          </div>
          <div className="camera-toggle">
            <button
              type="button"
              className={`secondary-btn ${stageView === "camera" ? "is-active" : ""}`}
              onClick={() => setStageView("camera")}
            >
              Show Camera
            </button>
            <button
              type="button"
              className={`secondary-btn ${stageView === "tracked" ? "is-active" : ""}`}
              onClick={() => setStageView("tracked")}
              disabled={!frameUrl}
            >
              Show Tracker Frame
            </button>
          </div>
          {canDefineRoi && (
            <button type="button" className="secondary-btn" onClick={beginRoiCapture}>
              {roiCaptureMode ? "Drawing ROI…" : "Draw ROI"}
            </button>
          )}
          {roi && (
            <p className="roi-readout">
              ROI → x:{roi.x}, y:{roi.y}, w:{roi.width}, h:{roi.height}
            </p>
          )}
        </article>

        <article>
          <h2>Tracker Controls</h2>
          <form className="track-form" onSubmit={handleManualCapture}>
            <label>
              Tracker mode
              <select value={mode} onChange={(event) => handleModeChange(event.target.value as Mode)}>
                <option value="aruco">Marker (ArUco)</option>
                <option value="of">Marker-less (Optical Flow)</option>
                <option value="sam2">SAM2 Segmentation</option>
              </select>
            </label>

            {mode === "aruco" && (
              <label>
                Marker ID (optional)
                <input
                  type="number"
                  min="0"
                  value={markerId}
                  onChange={(event) => setMarkerId(event.target.value)}
                />
              </label>
            )}

            {mode === "sam2" && (
              <label>
                SAM2 masks (.npz)
                <input type="file" accept=".npz" onChange={(event) => handleSam2Input(event.target.files?.[0] || null)} />
                <span className="field-hint">Up to {sam2LimitLabel}.</span>
              </label>
            )}

            <div className="button-row">
              <button type="button" className="primary-btn" onClick={startStreaming} disabled={sending || streaming}>
                {streaming ? "Streaming…" : "Start Tracking"}
              </button>
              <button type="button" className="secondary-btn" onClick={stopStreaming}>
                Stop Tracking
              </button>
              <button type="button" className="secondary-btn" onClick={resetSession}>
                Reset Session
              </button>
            </div>
            <button type="submit" className="ghost-btn">
              Capture Single Frame
            </button>
          </form>

          <div className="session-info">
            <p>
              <strong>Session ID:</strong> {sessionId ?? "—"}
            </p>
            <p>
              <strong>Streaming:</strong> {streaming ? "Yes" : "No"}
            </p>
            <p>
              <strong>Frame cap:</strong> {TRACKER_LIMITS.sessionMaxFrames} frames/session
            </p>
            <p>
              <strong>Idle timeout:</strong> ~{TRACKER_LIMITS.sessionTtlMinutes} min inactivity
            </p>
          </div>
        </article>

        <article>
          <h2>Tracker Output</h2>
          {frameUrl ? <img src={frameUrl} alt="Tracked frame" className="tracker-frame" /> : <p>No frames received yet.</p>}
          {metaEntries.length > 0 && (
            <div className="meta-panel">
              <h3>Metadata</h3>
              <ul>
                {metaEntries.map(([key, value]) => (
                  <li key={key}>
                    <strong>{key}:</strong> {JSON.stringify(value)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </article>
      </div>
    </section>
  );
}

export default ModuleFivePage;
