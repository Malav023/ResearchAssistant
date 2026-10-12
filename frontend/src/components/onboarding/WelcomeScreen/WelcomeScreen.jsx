import { useNavigate } from "react-router-dom";
import BentoCard from "../../common/BentoCard/BentoCard";
import Button from "../../common/Button/Button";
import styles from "./WelcomeScreen.module.css";

export default function WelcomeScreen() {
  const navigate = useNavigate();

  return (
    <div className={styles.wrap}>
      <BentoCard className={styles.card}>
        <div className={styles.badge}>Research Assistant</div>
        <h1 className={styles.title}>Explore research, and how it connects.</h1>
        <p className={styles.subtitle}>
          Search any topic to surface the most relevant papers, then dig into how they cite and relate
          to one another with an interactive citation graph — all in one place.
        </p>
        <div className={styles.actions}>
          <Button size="lg" onClick={() => navigate("/topics")}>
            Get started
          </Button>
        </div>
      </BentoCard>

      <div className={styles.grid}>
        <BentoCard className={styles.feature}>
          <div className={styles.featureTitle}>Fast paper discovery</div>
          <p className={styles.featureText}>
            Get the top 5 most relevant papers for any query, pulled from OpenAlex and enriched with
            Semantic Scholar citation data.
          </p>
        </BentoCard>
        <BentoCard className={styles.feature}>
          <div className={styles.featureTitle}>Visual citation graphs</div>
          <p className={styles.featureText}>
            See how papers cite and relate to each other in an interactive, explorable graph.
          </p>
        </BentoCard>
        <BentoCard className={styles.feature}>
          <div className={styles.featureTitle}>Your research history</div>
          <p className={styles.featureText}>
            Every search is saved to your account, so you can pick up right where you left off.
          </p>
        </BentoCard>
      </div>
    </div>
  );
}
