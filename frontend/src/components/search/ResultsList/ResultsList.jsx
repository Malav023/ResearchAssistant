import Button from "../../common/Button/Button";
import PaperCard from "../PaperCard/PaperCard";
import styles from "./ResultsList.module.css";

export default function ResultsList({ papers, query, loading, error, onSelectPaper, onOpenGraph }) {
  if (loading) {
    return (
      <div className={styles.state}>
        <span className={styles.spinner} />
        <span>Searching papers…</span>
      </div>
    );
  }

  if (error) {
    return <div className={`${styles.state} ${styles.error}`}>{error}</div>;
  }

  if (!papers || papers.length === 0) return null;

  return (
    <div className={styles.wrap}>
      <div className={styles.header}>
        <h2 className={styles.heading}>Top results for “{query}”</h2>
        <Button variant="secondary" size="sm" onClick={() => onOpenGraph?.(query)}>
          Citation graph
        </Button>
      </div>

      <div className={styles.grid}>
        {papers.map((paper) => (
          <PaperCard key={paper.id} paper={paper} onClick={onSelectPaper} />
        ))}
      </div>
    </div>
  );
}
