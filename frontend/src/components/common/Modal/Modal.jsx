import { useEffect, useRef } from "react";
import styles from "./Modal.module.css";

export default function Modal({ open, onClose, children, size = "md", ariaLabel }) {
  const contentRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleBackdropClick = (e) => {
    if (contentRef.current && !contentRef.current.contains(e.target)) {
      onClose?.();
    }
  };

  return (
    <div className={styles.backdrop} onMouseDown={handleBackdropClick}>
      <div
        ref={contentRef}
        className={`${styles.content} ${styles[size]}`}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
      >
        {children}
      </div>
    </div>
  );
}
