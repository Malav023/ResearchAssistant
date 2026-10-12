import { useAppState } from "../../../context/AppStateContext";
import styles from "./Sidebar.module.css";

export default function Sidebar({ sessions, onSelectSession }) {
  const { sidebarOpen, setSidebarOpen } = useAppState();

  // Nothing to show — don't render a sidebar shell at all.
  if (!sessions || sessions.length === 0) return null;

  return (
    <>
      {sidebarOpen && <div className={styles.scrim} onClick={() => setSidebarOpen(false)} />}
      <aside className={`${styles.sidebar} ${sidebarOpen ? styles.open : styles.closed}`}>
        <div className={styles.header}>
          <span className={styles.title}>Previous searches</span>
          <button className={styles.closeBtn} onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">
            ✕
          </button>
        </div>
        <div className={styles.list}>
          {sessions.map((s) => (
            <button
              key={s.id}
              className={styles.item}
              onClick={() => onSelectSession?.(s.query)}
              type="button"
            >
              <span className={styles.itemQuery}>{s.query}</span>
              <span className={styles.itemDate}>
                {new Date(s.created_at).toLocaleDateString()}
              </span>
            </button>
          ))}
        </div>
      </aside>
    </>
  );
}
