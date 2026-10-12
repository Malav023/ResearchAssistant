import { useState } from "react";
import styles from "./ChatBar.module.css";

export default function ChatBar({ onSearch, loading, initialValue = "" }) {
  const [value, setValue] = useState(initialValue);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim()) onSearch?.(value);
  };

  return (
    <form className={styles.bar} onSubmit={handleSubmit}>
      <input
        className={styles.input}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Search a research topic…"
        aria-label="Search a research topic"
      />
      <button
        type="submit"
        className={styles.submit}
        disabled={loading || !value.trim()}
        aria-label="Search"
      >
        {loading ? (
          <span className={styles.spinner} />
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M12 19V5M12 5L6 11M12 5L18 11"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </button>
    </form>
  );
}
