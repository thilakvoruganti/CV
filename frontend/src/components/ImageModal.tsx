import { ReactNode, useEffect } from "react";
import "./ImageModal.css";

interface ImageModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
}

export function ImageModal({ open, onClose, title, children }: ImageModalProps) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        <button className="modal-close" onClick={onClose} aria-label="Close preview">
          ×
        </button>
        {title && <h3>{title}</h3>}
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

export default ImageModal;
