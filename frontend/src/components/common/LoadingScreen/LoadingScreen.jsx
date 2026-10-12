import styles from "./LoadingScreen.module.css";

export default function LoadingScreen({ label = "Loading…", fullScreen = true }) {
  return (
    <div className={fullScreen ? styles.fullScreen : styles.inline} role="status" aria-live="polite">
      <div className={styles.mark}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
      <div className={styles.brand}>Research Assistant</div>
      <div className={styles.label}>{label}</div>
    </div>
  );
}
