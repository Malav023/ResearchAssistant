import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./PaperDetailPanel.module.css";

const MIN_WIDTH = 280;
const MAX_WIDTH = 520;
const DEFAULT_WIDTH = 380;

export default function PaperDetailPanel({ paper, onClose }) {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const draggingRef = useRef(false);

  const onMouseDown = useCallback(() => {
    draggingRef.current = true;
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!draggingRef.current) return;
      const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX));
      setWidth(next);
    };
    const onMouseUp = () => {
      draggingRef.current = false;
      document.body.style.userSelect = "";
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
    return () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!paper) return null;

  const doiUrl = paper.doi ? `https://doi.org/${paper.doi}` : "";

  return (
    <>
      <div className={styles.scrim} onClick={onClose} />
      <aside className={styles.panel} style={{ width }}>
        <div
          className={styles.handle}
          onMouseDown={onMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panel"
        />
        <div className={styles.header}>
          <span className={styles.eyebrow}>Paper detail</span>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close panel" type="button">
            ✕
          </button>
        </div>

        <div className={styles.content}>
          <h2 className={styles.title}>{paper.title}</h2>

          {paper.top_concept && <span className={styles.badge}>{paper.top_concept}</span>}

          <dl className={styles.kv}>
            <dt>First author</dt>
            <dd>{paper.first_author || "—"}</dd>
            <dt>Year</dt>
            <dd>{paper.year || "—"}</dd>
            <dt>Cited by</dt>
            <dd>{paper.citations.toLocaleString()}</dd>
          </dl>

          {doiUrl && (
            <a className={styles.link} href={doiUrl} target="_blank" rel="noopener noreferrer">
              View paper ↗
            </a>
          )}
        </div>
      </aside>
    </>
  );
}
