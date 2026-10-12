import { useState } from "react";
import { useNavigate } from "react-router-dom";
import BentoCard from "../../common/BentoCard/BentoCard";
import Button from "../../common/Button/Button";
import { useAppState } from "../../../context/AppStateContext";
import styles from "./TopicSelector.module.css";

const TOPICS = [
  "AI / Machine Learning",
  "Systems & Networking",
  "Biology & Health",
  "Physics",
  "Social Science",
  "Mathematics",
  "Climate & Earth Science",
  "Robotics",
];

export default function TopicSelector() {
  const [selected, setSelected] = useState([]);
  const { setPendingTopics } = useAppState();
  const navigate = useNavigate();

  const toggle = (topic) => {
    setSelected((prev) =>
      prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic]
    );
  };

  const handleContinue = () => {
    setPendingTopics(selected);
    navigate("/auth");
  };

  return (
    <div className={styles.wrap}>
      <BentoCard className={styles.card}>
        <h2 className={styles.title}>What are you researching?</h2>
        <p className={styles.subtitle}>
          Pick a few areas of interest — you can change these anytime. This just helps us tailor
          things for you.
        </p>

        <div className={styles.grid}>
          {TOPICS.map((topic) => {
            const active = selected.includes(topic);
            return (
              <button
                key={topic}
                type="button"
                className={`${styles.chip} ${active ? styles.chipActive : ""}`}
                onClick={() => toggle(topic)}
                aria-pressed={active}
              >
                {topic}
              </button>
            );
          })}
        </div>

        <div className={styles.actions}>
          <Button variant="ghost" onClick={() => navigate("/auth")}>
            Skip for now
          </Button>
          <Button onClick={handleContinue} disabled={selected.length === 0}>
            Continue
          </Button>
        </div>
      </BentoCard>
    </div>
  );
}
