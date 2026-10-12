import { useAppState } from "../../../context/AppStateContext";
import { useAuth } from "../../../hooks/useAuth";
import styles from "./AppShell.module.css";

export default function AppShell({ hasSidebar, children }) {
  const { sidebarOpen, toggleSidebar, activePaper, user } = useAppState();
  const { logout } = useAuth();

  const desktopShifted = hasSidebar && sidebarOpen;

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          {hasSidebar && (
            <button className={styles.iconBtn} onClick={toggleSidebar} aria-label="Toggle sidebar">
              ☰
            </button>
          )}
          <span className={styles.brand}>Research Assistant</span>
        </div>
        <div className={styles.headerRight}>
          {user && <span className={styles.userName}>{user.name}</span>}
          <button className={styles.logoutBtn} onClick={logout} type="button">
            Log out
          </button>
        </div>
      </header>

      <main
        className={`${styles.main} ${desktopShifted ? styles.mainShifted : ""} ${
          activePaper ? styles.mainDimmed : ""
        }`}
      >
        {children}
      </main>
    </div>
  );
}
