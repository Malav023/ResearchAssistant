import BentoCard from "../../common/BentoCard/BentoCard";
import styles from "./PaperCard.module.css";

export default function PaperCard({ paper, onClick }) {
  return (
    <BentoCard as="button" type="button" interactive className={styles.card} onClick={() => onClick?.(paper)}>
      <h3 className={styles.title}>{paper.title}</h3>
      <div className={styles.meta}>
        {paper.first_author && <span>{paper.first_author}</span>}
        {paper.year && <span>{paper.year}</span>}
        <span>{paper.citations.toLocaleString()} citations</span>
      </div>
      {paper.top_concept && <span className={styles.badge}>{paper.top_concept}</span>}
    </BentoCard>
  );
}
